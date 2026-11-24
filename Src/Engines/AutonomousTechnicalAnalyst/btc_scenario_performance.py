"""BTC scenario performance tracking.

Records confirmed monitoring observations, evaluates activated scenarios, and
tracks unconfirmed breakout near-misses separately. This module never trades.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
OBSERVATIONS_FILE = ROOT_DIR / "Data" / "Signals" / "btc_scenario_observations.json"
OUTCOMES_FILE = ROOT_DIR / "Data" / "Signals" / "btc_scenario_outcomes.json"
PERFORMANCE_FILE = ROOT_DIR / "Data" / "Signals" / "btc_performance_report.json"
HORIZONS = {"1H": 3600, "4H": 14400, "12H": 43200, "24H": 86400}
NEAR_MISS_HORIZONS = {"1H": 3600, "4H": 14400}
MAX_RECORDS = 20000


def utc_iso(timestamp):
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def read_store(path, default):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Performance store must be a JSON object: {path}")
    return data


def write_store(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def _scenario_snapshot(session, direction):
    scenario = session.get("scenarios", {}).get(direction, {})
    evaluation = session.get("latest_checks", {}).get(direction, {})
    checks = evaluation.get("checks", {})
    known_checks = [item for item in checks.values() if item.get("known")]
    confirmations = sum(1 for item in known_checks if item.get("met"))
    breakout = checks.get("4H_BREAKOUT", {})
    momentum_checks = [
        checks.get("15M_MOMENTUM", {}),
        checks.get("5M_MOMENTUM", {}),
    ]
    momentum_ready = all(
        item.get("known") and item.get("met") for item in momentum_checks
    )
    if scenario.get("status") == "ACTIVATED":
        classification = "ACTIVATED"
    elif breakout.get("known") and breakout.get("met") and confirmations < 3:
        classification = "UNCONFIRMED_BREAKOUT_NEAR_MISS"
    elif breakout.get("known") and not breakout.get("met") and momentum_ready:
        classification = "MOMENTUM_READY_BREAKOUT_PENDING"
    elif confirmations == 3:
        classification = "ALL_CHECKS_MET"
    else:
        classification = "WAITING"
    return {
        "status": scenario.get("status", "WAITING"),
        "classification": classification,
        "confirmations": confirmations,
        "known_conditions": len(known_checks),
        "checks": checks,
        "near_miss": bool(
            scenario.get("status") != "ACTIVATED"
            and breakout.get("known")
            and breakout.get("met")
            and not all(item.get("known") and item.get("met") for item in momentum_checks)
        ),
    }


def record_observation(session, observations_path=OBSERVATIONS_FILE, now_timestamp=None):
    """Append one deduplicated observation for each new confirmed 5M candle."""
    if session.get("status") != "ACTIVE":
        return False
    checks = session.get("latest_checks")
    if not isinstance(checks, dict):
        return False
    observed_timestamp = max(
        (int(item.get("observed_timestamp") or 0) for item in checks.values()),
        default=0,
    )
    observed_price = next(
        (item.get("observed_price") for item in checks.values()
         if item.get("observed_price") is not None),
        None,
    )
    if observed_timestamp <= 0 or observed_price is None:
        return False
    store = read_store(observations_path, {
        "schema_version": "1.0", "symbol": "BTC-USDT", "records": []
    })
    records = store.setdefault("records", [])
    record_id = f"{session['analysis_id']}:{observed_timestamp}"
    if any(item.get("record_id") == record_id for item in records):
        return False
    record = {
        "record_id": record_id,
        "analysis_id": session["analysis_id"],
        "observed_at": utc_iso(observed_timestamp),
        "observed_timestamp": observed_timestamp,
        "recorded_at": utc_iso(now_timestamp if now_timestamp is not None else observed_timestamp),
        "price": float(observed_price),
        "scenarios": {
            direction: _scenario_snapshot(session, direction)
            for direction in ("LONG", "SHORT")
        },
    }
    records.append(record)
    store["records"] = records[-MAX_RECORDS:]
    store["updated_at"] = utc_iso(now_timestamp if now_timestamp is not None else observed_timestamp)
    write_store(observations_path, store)
    return True


def _new_outcome(session, direction, kind, entry_timestamp, entry_price, level):
    horizons = NEAR_MISS_HORIZONS if kind == "UNCONFIRMED_NEAR_MISS" else HORIZONS
    return {
        "outcome_id": f"{session['analysis_id']}:{direction}:{kind}:{int(entry_timestamp)}",
        "analysis_id": session["analysis_id"],
        "direction": direction,
        "kind": kind,
        "entry_timestamp": int(entry_timestamp),
        "entry_at": utc_iso(entry_timestamp),
        "entry_price": float(entry_price),
        "breakout_level": float(level) if level is not None else None,
        "windows": {
            label: {
                "target_timestamp": int(entry_timestamp + seconds),
                "max_favorable_price": float(entry_price),
                "max_adverse_price": float(entry_price),
                "last_candle_timestamp": None,
                "resolved": False,
            }
            for label, seconds in horizons.items()
        },
        "breakout_assessment": (
            {"status": "PENDING", "checked_through": None}
            if kind == "ACTIVATED" else None
        ),
    }


def _update_window(outcome, window, candle):
    entry = outcome["entry_price"]
    if outcome["direction"] == "LONG":
        window["max_favorable_price"] = max(window["max_favorable_price"], candle["high"])
        window["max_adverse_price"] = min(window["max_adverse_price"], candle["low"])
        favorable = (window["max_favorable_price"] - entry) / entry * 100.0
        adverse = (entry - window["max_adverse_price"]) / entry * 100.0
        forward = (candle["close"] - entry) / entry * 100.0
    else:
        window["max_favorable_price"] = min(window["max_favorable_price"], candle["low"])
        window["max_adverse_price"] = max(window["max_adverse_price"], candle["high"])
        favorable = (entry - window["max_favorable_price"]) / entry * 100.0
        adverse = (window["max_adverse_price"] - entry) / entry * 100.0
        forward = (entry - candle["close"]) / entry * 100.0
    window["max_favorable_excursion_pct"] = max(0.0, favorable)
    window["max_adverse_excursion_pct"] = max(0.0, adverse)
    window["last_candle_timestamp"] = int(candle["timestamp"])
    close_timestamp = int(candle["timestamp"]) + 300
    if close_timestamp >= int(window["target_timestamp"]):
        window["forward_close"] = float(candle["close"])
        window["forward_return_pct"] = forward
        window["resolved_at"] = utc_iso(close_timestamp)
        window["resolved"] = True


def update_outcomes(session, candles_by_tf, outcomes_path=OUTCOMES_FILE, now_timestamp=None):
    """Track MFE/MAE and forward returns for activations and near-miss candidates."""
    store = read_store(outcomes_path, {
        "schema_version": "1.0", "symbol": "BTC-USDT", "records": []
    })
    records = store.setdefault("records", [])
    existing = {item.get("outcome_id") for item in records}

    for direction in ("LONG", "SHORT"):
        scenario = session.get("scenarios", {}).get(direction, {})
        activation = scenario.get("activation")
        if scenario.get("status") == "ACTIVATED" and activation:
            entry_time = int(activation.get("timestamp_unix") or 0)
            entry_price = activation.get("price")
            level = activation.get("checks", {}).get("4H_BREAKOUT", {}).get("level")
            if entry_time > 0 and entry_price is not None:
                candidate = _new_outcome(session, direction, "ACTIVATED", entry_time, entry_price, level)
                if candidate["outcome_id"] not in existing:
                    records.append(candidate)
                    existing.add(candidate["outcome_id"])

        evaluation = session.get("latest_checks", {}).get(direction, {})
        checks = evaluation.get("checks", {})
        breakout = checks.get("4H_BREAKOUT", {})
        momentum = [checks.get("15M_MOMENTUM", {}), checks.get("5M_MOMENTUM", {})]
        observed_time = int(evaluation.get("observed_timestamp") or 0)
        observed_price = evaluation.get("observed_price")
        if (
            session.get("status") == "ACTIVE"
            and scenario.get("status") != "ACTIVATED"
            and breakout.get("known") and breakout.get("met")
            and not all(item.get("known") and item.get("met") for item in momentum)
            and observed_time > 0 and observed_price is not None
        ):
            candidate = _new_outcome(
                session, direction, "UNCONFIRMED_NEAR_MISS", observed_time,
                observed_price, breakout.get("level"),
            )
            event_timestamp = int(breakout.get("candle_close_timestamp") or observed_time)
            candidate["outcome_id"] = (
                f"{session['analysis_id']}:{direction}:UNCONFIRMED_NEAR_MISS:{event_timestamp}"
            )
            if candidate["outcome_id"] not in existing:
                records.append(candidate)
                existing.add(candidate["outcome_id"])

    five_minute = candles_by_tf.get("5M", [])
    four_hour = candles_by_tf.get("4H", [])
    for outcome in records:
        if outcome.get("windows") and not all(
            item.get("resolved") for item in outcome["windows"].values()
        ):
            for window in outcome["windows"].values():
                if window.get("resolved"):
                    continue
                last_seen = window.get("last_candle_timestamp")
                for candle in five_minute:
                    candle_open = int(candle["timestamp"])
                    candle_close = candle_open + 300
                    if candle_close <= outcome["entry_timestamp"]:
                        continue
                    if last_seen is not None and candle_open <= int(last_seen):
                        continue
                    _update_window(outcome, window, candle)
                    if window.get("resolved"):
                        break

        assessment = outcome.get("breakout_assessment")
        if assessment and assessment.get("status") == "PENDING":
            entry_time = int(outcome["entry_timestamp"])
            target_time = entry_time + 14400
            last_seen = assessment.get("checked_through")
            for candle in four_hour:
                candle_close = int(candle["timestamp"]) + 14400
                if candle_close <= entry_time or candle_close > target_time:
                    continue
                if last_seen is not None and candle_close <= int(last_seen):
                    continue
                level = outcome.get("breakout_level")
                failed = (
                    candle["close"] <= level if outcome["direction"] == "LONG"
                    else candle["close"] >= level
                )
                assessment["checked_through"] = candle_close
                if failed:
                    assessment["status"] = "FAILED_RECLAIM"
                    assessment["failed_at"] = utc_iso(candle_close)
                    break
            if (
                assessment["status"] == "PENDING"
                and assessment.get("checked_through") is not None
                and int(assessment["checked_through"]) >= target_time
            ):
                assessment["status"] = "HELD_THROUGH_4H"

    store["records"] = records[-MAX_RECORDS:]
    now = now_timestamp if now_timestamp is not None else max(
        [int(c["timestamp"]) + 300 for c in five_minute] or [0]
    )
    store["updated_at"] = utc_iso(now) if now else None
    write_store(outcomes_path, store)
    return store


def _mean(values):
    return sum(values) / len(values) if values else None


def _condition_statistics(observation_records, direction, condition):
    known = []
    for observation in observation_records:
        snapshot = observation.get("scenarios", {}).get(direction, {})
        check = snapshot.get("checks", {}).get(condition, {})
        if check.get("known"):
            known.append(bool(check.get("met")))
    passed = sum(1 for value in known if value)
    return {
        "known_count": len(known),
        "passed_count": passed,
        "failed_count": len(known) - passed,
        "pass_rate_pct": round(passed / len(known) * 100.0, 4) if known else None,
    }


def _horizon_statistics(records, horizon):
    resolved = [
        item["windows"][horizon]
        for item in records
        if horizon in item.get("windows", {}) and item["windows"][horizon].get("resolved")
    ]
    returns = [item["forward_return_pct"] for item in resolved if "forward_return_pct" in item]
    mfe = [item["max_favorable_excursion_pct"] for item in resolved if "max_favorable_excursion_pct" in item]
    mae = [item["max_adverse_excursion_pct"] for item in resolved if "max_adverse_excursion_pct" in item]
    return {
        "resolved_count": len(resolved),
        "positive_forward_return_count": sum(1 for value in returns if value > 0),
        "positive_forward_return_rate_pct": (
            round(sum(1 for value in returns if value > 0) / len(returns) * 100.0, 4)
            if returns else None
        ),
        "average_forward_return_pct": _mean(returns),
        "average_mfe_pct": _mean(mfe),
        "average_mae_pct": _mean(mae),
    }


def build_performance_report(observations, outcomes, generated_at=None):
    observation_records = observations.get("records", [])
    outcome_records = outcomes.get("records", [])
    report = {
        "schema_version": "2.0",
        "symbol": "BTC-USDT",
        "generated_at": utc_iso(
            generated_at if generated_at is not None
            else datetime.now(timezone.utc).timestamp()
        ),
        "methodology": {
            "excursions": "MFE and MAE use confirmed 5-minute candle highs/lows after the observation or activation price.",
            "forward_return": "Direction-adjusted return at the first confirmed 5-minute close at or after each horizon.",
            "failed_breakout": "A failed 4H breakout is recorded only if a confirmed 4H candle closes back through the frozen breakout level within 4 hours.",
            "near_miss": "A frozen 4H breakout is met but one or both 15M/5M momentum confirmations are missing; this is a candidate, not an activated signal.",
            "momentum_ready_breakout_pending": "Both 15M and 5M momentum checks pass while the frozen 4H breakout check does not; this is a pending setup candidate, not a confirmed missed trade.",
            "condition_rates": "Pass rates use only observations where that condition was known; unknown/stale conditions are excluded from the denominator.",
            "activation_rate": "Activations per 100 monitoring observations is descriptive only and is not a probability of trade success.",
            "not_a_trading_backtest": True,
        },
        "observations_count": len(observation_records),
        "near_miss_observations": {
            direction: sum(
                1 for item in observation_records
                if item.get("scenarios", {}).get(direction, {}).get("near_miss")
            )
            for direction in ("LONG", "SHORT")
        },
        "scenarios": {},
    }
    for direction in ("LONG", "SHORT"):
        direction_observations = [
            item for item in observation_records
            if direction in item.get("scenarios", {})
        ]
        snapshots = [
            item["scenarios"][direction] for item in direction_observations
        ]
        direction_records = [
            item for item in outcome_records if item.get("direction") == direction
        ]
        activated = [item for item in direction_records if item.get("kind") == "ACTIVATED"]
        near_misses = [
            item for item in direction_records
            if item.get("kind") == "UNCONFIRMED_NEAR_MISS"
        ]
        momentum_ready = sum(
            1 for item in snapshots
            if item.get("classification") == "MOMENTUM_READY_BREAKOUT_PENDING"
        )
        near_miss_observations = sum(
            1 for item in snapshots
            if item.get("classification") == "UNCONFIRMED_BREAKOUT_NEAR_MISS"
        )
        activated_observations = sum(
            1 for item in snapshots
            if item.get("status") == "ACTIVATED"
            or item.get("classification") == "ACTIVATED"
        )
        condition_statistics = {
            "4H_BREAKOUT": _condition_statistics(
                observation_records, direction, "4H_BREAKOUT"
            ),
            "15M_MOMENTUM": _condition_statistics(
                observation_records, direction, "15M_MOMENTUM"
            ),
            "5M_MOMENTUM": _condition_statistics(
                observation_records, direction, "5M_MOMENTUM"
            ),
        }
        scenario_summary = {
            "monitoring_observations": len(direction_observations),
            "activated_count": len(activated),
            "activated_observation_count": activated_observations,
            "activations_per_100_observations": (
                round(len(activated) / len(direction_observations) * 100.0, 4)
                if direction_observations else None
            ),
            "near_miss_candidate_count": len(near_misses),
            "near_miss_observation_count": near_miss_observations,
            "momentum_ready_breakout_pending_observation_count": momentum_ready,
            "potential_missed_opportunity_observation_count": near_miss_observations + momentum_ready,
            "failed_breakout_count": sum(
                1 for item in activated
                if item.get("breakout_assessment", {}).get("status") == "FAILED_RECLAIM"
            ),
            "breakout_held_4h_count": sum(
                1 for item in activated
                if item.get("breakout_assessment", {}).get("status") == "HELD_THROUGH_4H"
            ),
            "breakout_assessment_pending_count": sum(
                1 for item in activated
                if item.get("breakout_assessment", {}).get("status") == "PENDING"
            ),
            "condition_statistics": condition_statistics,
            "horizons": {
                horizon: _horizon_statistics(activated, horizon)
                for horizon in HORIZONS
            },
            "near_miss_horizons": {
                horizon: _horizon_statistics(near_misses, horizon)
                for horizon in NEAR_MISS_HORIZONS
            },
        }
        report["scenarios"][direction] = scenario_summary

    long_summary = report["scenarios"]["LONG"]
    short_summary = report["scenarios"]["SHORT"]
    comparison = {}
    for horizon in HORIZONS:
        long_return = long_summary["horizons"][horizon]["average_forward_return_pct"]
        short_return = short_summary["horizons"][horizon]["average_forward_return_pct"]
        if long_return is None or short_return is None:
            preferred = "INSUFFICIENT_DATA"
        elif long_return > short_return:
            preferred = "LONG"
        elif short_return > long_return:
            preferred = "SHORT"
        else:
            preferred = "TIE"
        comparison[horizon] = {
            "long_average_forward_return_pct": long_return,
            "short_average_forward_return_pct": short_return,
            "higher_average_forward_return_direction": preferred,
        }
    report["direction_comparison"] = {
        "higher_activation_count": (
            "LONG" if long_summary["activated_count"] > short_summary["activated_count"]
            else "SHORT" if short_summary["activated_count"] > long_summary["activated_count"]
            else "TIE"
        ),
        "higher_4h_positive_return_rate": comparison["4H"][
            "higher_average_forward_return_direction"
        ],
        "by_horizon": comparison,
        "warning": "Do not rank LONG versus SHORT until each direction has a sufficient resolved sample.",
    }
    return report

def refresh_performance_report(observations_path=OBSERVATIONS_FILE, outcomes_path=OUTCOMES_FILE,
                               report_path=PERFORMANCE_FILE, generated_at=None):
    observations = read_store(observations_path, {"records": []})
    outcomes = read_store(outcomes_path, {"records": []})
    report = build_performance_report(observations, outcomes, generated_at)
    write_store(report_path, report)
    return report
