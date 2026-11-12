
import sys
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings

from Src.Data.okx_client import (
    fetch_historical_candles,
    fetch_new_candles,
)

from Src.Data.archive import (
    save_candles,
    get_latest_timestamp,
)

from Src.Analysis.daily_structure import (
    analyze_daily_structure,
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)

from Src.Analysis.market_direction import (
    determine_market_direction,
)

from Src.Analysis.four_hour_setup import (
    generate_four_hour_setup,
)

from Src.Analysis.entry_confirmation import (
    confirm_lower_timeframe_entry,
)


MARKET = "SPOT"
SYMBOL = "BTC-USDT"

RUN_HISTORICAL_CONFLUENCE_AUDIT = True

TIMEFRAME_CONFIG = {
    "1D": {
        "target": 365,
        "interval_ms": 86400000,
        "filename": "BTC_USDT_SPOT_1D.csv",
    },
    "4H": {
        "target": 540,
        "interval_ms": 14400000,
        "filename": "BTC_USDT_SPOT_4H.csv",
    },
    "1H": {
        "target": 720,
        "interval_ms": 3600000,
        "filename": "BTC_USDT_SPOT_1H.csv",
    },
    "15m": {
        "target": 960,
        "interval_ms": 900000,
        "filename": "BTC_USDT_SPOT_15m.csv",
    },
    "5m": {
        "target": 1016,
        "interval_ms": 300000,
        "filename": "BTC_USDT_SPOT_5m.csv",
    },
}


def get_archive_path(filename):
    return PROJECT_ROOT / "Data" / "Archive" / filename


def count_archive_rows(filename):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        return 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        return sum(1 for row in reader)


def initialize_archive(timeframe, config):
    filename = config["filename"]
    target = config["target"]
    archive_path = get_archive_path(filename)

    if timeframe == "5m" and not archive_path.exists():
        raise RuntimeError(
            "Existing 5m archive is missing. "
            "Automatic recreation is disabled."
        )

    current_count = count_archive_rows(filename)

    if current_count >= target:
        print(
            f"{timeframe}: Existing archive contains "
            f"{current_count} candles. Initial seed not required.",
            flush=True,
        )
        return

    print(
        f"{timeframe}: Initializing historical archive. "
        f"Target: {target} candles.",
        flush=True,
    )

    candles = fetch_historical_candles(
        market=MARKET,
        timeframe=timeframe,
        total_candles=target,
        page_size=100,
    )

    if len(candles) != target:
        raise RuntimeError(
            f"{timeframe}: Historical seed incomplete. "
            f"Expected {target}, received {len(candles)}. "
            "Archive was not updated."
        )

    result = save_candles(candles, filename)

    print(
        f"{timeframe}: Historical seed saved. "
        f"New candles: {result['new_candles']}, "
        f"total candles: {result['total_candles']}",
        flush=True,
    )


def update_archive(timeframe, config):
    filename = config["filename"]

    last_timestamp = get_latest_timestamp(filename)

    if last_timestamp is None:
        raise RuntimeError(
            f"{timeframe}: Archive has no latest timestamp."
        )

    last_time = datetime.fromtimestamp(
        last_timestamp / 1000,
        tz=timezone.utc,
    )

    print(f"\n{timeframe} INCREMENTAL UPDATE", flush=True)
    print("============================", flush=True)
    print(f"Last Archived UTC: {last_time.isoformat()}", flush=True)

    candles = fetch_new_candles(
        market=MARKET,
        timeframe=timeframe,
        last_timestamp=last_timestamp,
        page_size=100,
    )

    print(f"New Candles Retrieved: {len(candles)}", flush=True)

    if candles:
        result = save_candles(candles, filename)

        print(f"New Candles Saved: {result['new_candles']}", flush=True)
        print(f"Total Archived Candles: {result['total_candles']}", flush=True)

        newest_time = datetime.fromtimestamp(
            candles[-1]["timestamp"] / 1000,
            tz=timezone.utc,
        )

        print(f"Newest Candle UTC: {newest_time.isoformat()}", flush=True)

    else:
        print("No new completed candles found.", flush=True)
        print("Archive remains unchanged.", flush=True)


def audit_archive(filename, interval_ms):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        raise RuntimeError(
            f"Archive file not found: {filename}"
        )

    timestamps = []
    invalid_rows = 0
    misaligned_timestamps = 0
    total_rows = 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            try:
                timestamp = int(row["timestamp"])

                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
                volume = float(row["volume"])

                if (
                    min(
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                    ) <= 0
                    or volume < 0
                    or high_price < max(
                        open_price,
                        close_price,
                        low_price,
                    )
                    or low_price > min(
                        open_price,
                        close_price,
                        high_price,
                    )
                ):
                    invalid_rows += 1

                timestamps.append(timestamp)

                if timestamp % interval_ms != 0:
                    misaligned_timestamps += 1

            except (ValueError, KeyError, TypeError):
                invalid_rows += 1

    if not timestamps:
        raise RuntimeError(
            f"{filename}: Archive contains no valid timestamps."
        )

    unique_timestamps = sorted(set(timestamps))

    duplicate_count = (
        len(timestamps) - len(unique_timestamps)
    )

    missing_candles = 0
    gap_count = 0

    for index in range(1, len(unique_timestamps)):
        difference = (
            unique_timestamps[index]
            - unique_timestamps[index - 1]
        )

        if difference > interval_ms:
            missing = (
                difference // interval_ms
            ) - 1

            missing_candles += missing
            gap_count += 1

        elif difference < interval_ms:
            invalid_rows += 1

    first_time = datetime.fromtimestamp(
        unique_timestamps[0] / 1000,
        tz=timezone.utc,
    )

    last_time = datetime.fromtimestamp(
        unique_timestamps[-1] / 1000,
        tz=timezone.utc,
    )

    expected_count = (
        (
            unique_timestamps[-1]
            - unique_timestamps[0]
        )
        // interval_ms
    ) + 1

    continuous = (
        len(unique_timestamps) == expected_count
    )

    passed = (
        duplicate_count == 0
        and missing_candles == 0
        and invalid_rows == 0
        and misaligned_timestamps == 0
        and continuous
    )

    print("\nARCHIVE AUDIT REPORT", flush=True)
    print("====================", flush=True)
    print(f"Archive File: {filename}", flush=True)
    print(f"Total Rows: {total_rows}", flush=True)
    print(f"Unique Candles: {len(unique_timestamps)}", flush=True)
    print(f"Duplicate Rows: {duplicate_count}", flush=True)
    print(f"Missing Candles: {missing_candles}", flush=True)
    print(f"Gap Segments: {gap_count}", flush=True)
    print(f"Invalid Rows: {invalid_rows}", flush=True)
    print(f"Misaligned Timestamps: {misaligned_timestamps}", flush=True)
    print(f"First Candle UTC: {first_time.isoformat()}", flush=True)
    print(f"Last Candle UTC: {last_time.isoformat()}", flush=True)
    print(f"Expected Continuous Count: {expected_count}", flush=True)
    print(f"Continuous Timeline: {continuous}", flush=True)
    print(
        f"AUDIT STATUS: "
        f"{'PASS' if passed else 'ISSUES DETECTED'}",
        flush=True,
    )

    return passed


def load_structure_candles(filename, limit):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        raise RuntimeError(
            f"Structure archive not found: {filename}"
        )

    candles = []

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                candle = {
                    "timestamp": int(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }

                candles.append(candle)

            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"Invalid candle in {filename}: {error}"
                )

    candles.sort(
        key=lambda candle: candle["timestamp"]
    )

    if len(candles) < limit:
        raise RuntimeError(
            f"{filename}: Insufficient candles. "
            f"Required {limit}, found {len(candles)}."
        )

    return candles[-limit:]


def run_market_direction_analysis():
    print("\nREAL MARKET DIRECTION ANALYSIS", flush=True)
    print("==============================", flush=True)

    daily_candles = load_structure_candles(
        TIMEFRAME_CONFIG["1D"]["filename"],
        90,
    )

    four_hour_candles = load_structure_candles(
        TIMEFRAME_CONFIG["4H"]["filename"],
        540,
    )

    print(f"Daily Candles: {len(daily_candles)}", flush=True)
    print(f"4H Candles: {len(four_hour_candles)}", flush=True)

    daily_result = analyze_daily_structure(
        daily_candles,
        timeframe="1D",
    )

    four_hour_result = analyze_daily_structure(
        four_hour_candles,
        timeframe="4H",
    )

    decision = determine_market_direction(
        daily_result,
        four_hour_result,
    )

    print("\nDAILY MARKET STRUCTURE", flush=True)
    print("======================", flush=True)
    print("Trend:", decision["daily_bias"], flush=True)
    print("Event Bias:", decision["daily_event_bias"], flush=True)
    print("Reason:", decision["daily_structure_reason"], flush=True)

    print("\n4H MARKET STRUCTURE", flush=True)
    print("==================", flush=True)
    print("Trend:", decision["four_hour_bias"], flush=True)
    print("Event Bias:", decision["four_hour_event_bias"], flush=True)
    print("Reason:", decision["four_hour_structure_reason"], flush=True)

    print("\nFINAL MARKET DIRECTION", flush=True)
    print("======================", flush=True)
    print("Allowed Direction:", decision["allowed_direction"], flush=True)
    print("Status:", decision["status"], flush=True)
    print("Reason:", decision["reason"], flush=True)
    print(
        "Trade Directions:",
        decision["trade_directions_to_evaluate"],
        flush=True,
    )

    return decision, four_hour_candles


def run_historical_confluence_audit(candles):
    print("\n" + "=" * 72, flush=True)
    print("HISTORICAL 4H CONFLUENCE AUDIT", flush=True)
    print("=" * 72, flush=True)
    print("Symbol:", SYMBOL, flush=True)
    print("Timeframe: 4H", flush=True)
    print("Method: Candle-by-Candle Event and Setup Audit", flush=True)
    print("Lookahead: DISABLED", flush=True)
    print("Directions: LONG and SHORT", flush=True)
    print("Live Trading: DISABLED", flush=True)
    print("Archive Modification: NO", flush=True)

    if not isinstance(candles, (list, tuple)) or len(candles) < 100:
        raise RuntimeError(
            "Insufficient candles for historical confluence audit."
        )

    validated_candles = validate_candles(list(candles))

    if not isinstance(validated_candles, list):
        raise RuntimeError(
            "Historical audit received invalid validated candles."
        )

    candle_duration = infer_candle_duration(validated_candles)

    if not isinstance(candle_duration, (int, float)) or candle_duration <= 0:
        raise RuntimeError(
            "Historical audit could not determine candle duration."
        )

    total_prefixes = len(validated_candles) - 99
    total_engine_evaluations = 0

    discovered_events = {}
    discovered_setups = {}
    evaluation_records = []
    engine_errors = []
    structure_errors = []

    status_counts = {}
    direction_counts = {
        "LONG": 0,
        "SHORT": 0,
    }

    print("\nAUDIT INITIALIZATION", flush=True)
    print("Historical Candles:", len(validated_candles), flush=True)
    print("Candle Duration:", candle_duration, flush=True)
    print("Evaluated Prefixes:", total_prefixes, flush=True)
    print("Expected Engine Evaluations:", total_prefixes * 2, flush=True)

    for end_index in range(100, len(validated_candles) + 1):
        historical_candles = validated_candles[:end_index]

        current_candle = historical_candles[-1]
        current_timestamp = current_candle["timestamp"]
        current_candle_close = current_timestamp + candle_duration

        prefix_index = end_index - 1

        prefix_events = []
        prefix_structure_error = None

        try:
            swings = detect_swings(
                historical_candles,
                left_bars=3,
                right_bars=3,
                candle_duration=candle_duration,
            )

            structure_events = analyze_structure_events(
                historical_candles,
                swings,
                candle_duration,
            )

            if isinstance(structure_events, list):
                prefix_events = structure_events

        except Exception as error:
            prefix_structure_error = {
                "candle_index": prefix_index,
                "timestamp": current_timestamp,
                "error_type": type(error).__name__,
                "error": str(error),
            }

            structure_errors.append(prefix_structure_error)

        for event in prefix_events:
            if not isinstance(event, dict):
                continue

            event_type = event.get("event")
            event_direction = event.get("direction")
            event_index = event.get("index")
            event_timestamp = event.get("timestamp")
            available_timestamp = event.get("available_timestamp")
            event_level = event.get("level")

            if event_type not in (
                "BOS_BULLISH",
                "BOS_BEARISH",
                "CHOCH_BULLISH",
                "CHOCH_BEARISH",
            ):
                continue

            if event_direction not in ("BULLISH", "BEARISH"):
                continue

            if available_timestamp is None:
                continue

            try:
                available_timestamp = int(available_timestamp)
            except (ValueError, TypeError, OverflowError):
                continue

            if available_timestamp > current_candle_close:
                continue

            if event_index is None:
                continue

            try:
                event_index = int(event_index)
            except (ValueError, TypeError, OverflowError):
                continue

            if event_index < 0 or event_index >= len(historical_candles):
                continue

            event_key = (
                event_type,
                event_direction,
                event_timestamp,
                event_level,
                event_index,
            )

            previous_event = discovered_events.get(event_key)

            if previous_event is None:
                discovered_events[event_key] = {
                    "event_type": event_type,
                    "direction": event_direction,
                    "index": event_index,
                    "timestamp": event_timestamp,
                    "available_timestamp": available_timestamp,
                    "level": event_level,
                    "first_seen_prefix": prefix_index,
                    "last_seen_prefix": prefix_index,
                    "observations": 1,
                }
            else:
                previous_event["last_seen_prefix"] = prefix_index
                previous_event["observations"] += 1

        for direction in ("LONG", "SHORT"):
            total_engine_evaluations += 1

            matching_events = [
                event
                for event in prefix_events
                if isinstance(event, dict)
                and event.get("direction") == (
                    "BULLISH" if direction == "LONG" else "BEARISH"
                )
                and event.get("available_timestamp") is not None
                and event.get("available_timestamp") <= current_candle_close
            ]

            latest_matching_event = None

            if matching_events:
                latest_matching_event = max(
                    matching_events,
                    key=lambda item: (
                        item.get("available_timestamp", 0),
                        item.get("index", 0),
                    ),
                )

            evaluation_record = {
                "prefix_index": prefix_index,
                "prefix_timestamp": current_timestamp,
                "direction": direction,
                "result_status": "NOT_EVALUATED",
                "reason": None,
                "setup_id": None,
                "setup_status": None,
                "structure_event": (
                    latest_matching_event.get("event")
                    if latest_matching_event
                    else None
                ),
                "structure_event_timestamp": (
                    latest_matching_event.get("timestamp")
                    if latest_matching_event
                    else None
                ),
                "diagnostics": None,
                "error": None,
            }

            try:
                result = generate_four_hour_setup(
                    candles=historical_candles,
                    allowed_direction=direction,
                    setup_expiry_candles=24,
                    ob_lookback=10,
                    fvg_lookback=10,
                    risk_reward=2.0,
                )

                if not isinstance(result, dict):
                    evaluation_record["result_status"] = "INVALID_RESULT"
                    evaluation_record["reason"] = (
                        "Engine returned a non-dictionary result."
                    )

                else:
                    setup = result.get("setup")
                    diagnostics = result.get("diagnostics")

                    evaluation_record["result_status"] = result.get(
                        "status",
                        "UNKNOWN",
                    )

                    evaluation_record["reason"] = result.get("reason")
                    evaluation_record["diagnostics"] = diagnostics

                    if isinstance(setup, dict):
                        setup_id = setup.get("setup_id")

                        evaluation_record["setup_id"] = setup_id
                        evaluation_record["setup_status"] = setup.get(
                            "status",
                            result.get("status"),
                        )

                        if setup_id:
                            previous_setup = discovered_setups.get(setup_id)

                            if previous_setup is None:
                                discovered_setups[setup_id] = {
                                    "setup_id": setup_id,
                                    "direction": setup.get("direction", direction),
                                    "status": setup.get(
                                        "status",
                                        result.get("status", "UNKNOWN"),
                                    ),
                                    "structure_event": setup.get("structure_event"),
                                    "structure_event_timestamp": setup.get(
                                        "structure_event_timestamp"
                                    ),
                                    "available_timestamp": setup.get(
                                        "available_timestamp"
                                    ),
                                    "expiry_timestamp": setup.get(
                                        "expiry_timestamp"
                                    ),
                                    "entry_zone": setup.get("entry_zone"),
                                    "order_block": setup.get("order_block"),
                                    "fvg": setup.get("fvg"),
                                    "zone_overlap": setup.get("zone_overlap"),
                                    "stop_loss": setup.get("stop_loss"),
                                    "take_profit": setup.get("take_profit"),
                                    "reason": result.get("reason"),
                                    "first_seen_prefix": prefix_index,
                                    "last_seen_prefix": prefix_index,
                                    "observations": 1,
                                }
                            else:
                                previous_setup["last_seen_prefix"] = prefix_index
                                previous_setup["observations"] += 1
                                previous_setup["status"] = setup.get(
                                    "status",
                                    result.get("status", "UNKNOWN"),
                                )

                    else:
                        evaluation_record["setup_status"] = "NO_SETUP"

                status_key = str(
                    evaluation_record["result_status"]
                )

                status_counts[status_key] = (
                    status_counts.get(status_key, 0) + 1
                )

                direction_counts[direction] += 1

            except Exception as error:
                evaluation_record["result_status"] = "ENGINE_ERROR"
                evaluation_record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

                engine_errors.append({
                    "prefix_index": prefix_index,
                    "timestamp": current_timestamp,
                    "direction": direction,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })

                status_counts["ENGINE_ERROR"] = (
                    status_counts.get("ENGINE_ERROR", 0) + 1
                )

                direction_counts[direction] += 1

            evaluation_records.append(evaluation_record)

        completed_prefixes = end_index - 99

        if completed_prefixes % 25 == 0 or completed_prefixes == total_prefixes:
            print(
                "Progress:",
                completed_prefixes,
                "/",
                total_prefixes,
                "| Engine Evaluations:",
                total_engine_evaluations,
                "| Structure Events:",
                len(discovered_events),
                "| Unique Setups:",
                len(discovered_setups),
                "| Engine Errors:",
                len(engine_errors),
                "| Structure Errors:",
                len(structure_errors),
                flush=True,
            )

    event_list = list(discovered_events.values())
    setup_list = list(discovered_setups.values())

    event_type_counts = {
        "BOS_BULLISH": 0,
        "BOS_BEARISH": 0,
        "CHOCH_BULLISH": 0,
        "CHOCH_BEARISH": 0,
    }

    for event in event_list:
        event_type = event["event_type"]

        if event_type in event_type_counts:
            event_type_counts[event_type] += 1

    setup_status_counts = {}

    for setup in setup_list:
        status = str(setup.get("status", "UNKNOWN"))

        setup_status_counts[status] = (
            setup_status_counts.get(status, 0) + 1
        )

    print("\n" + "=" * 72, flush=True)
    print("HISTORICAL AUDIT SUMMARY", flush=True)
    print("=" * 72, flush=True)

    print("Historical Candles:", len(validated_candles), flush=True)
    print("Evaluated Candle Prefixes:", total_prefixes, flush=True)
    print("Engine Evaluations:", total_engine_evaluations, flush=True)
    print("Recorded Structure Events:", len(event_list), flush=True)
    print("Unique Setups Discovered:", len(setup_list), flush=True)
    print("Recorded Evaluation Results:", len(evaluation_records), flush=True)
    print("Engine Errors:", len(engine_errors), flush=True)
    print("Structure Analysis Errors:", len(structure_errors), flush=True)

    print("\nSTRUCTURE EVENT DISTRIBUTION", flush=True)
    print("-" * 72, flush=True)

    for event_type, count in event_type_counts.items():
        print(f"{event_type}: {count}", flush=True)

    print("\nSETUP STATUS DISTRIBUTION", flush=True)
    print("-" * 72, flush=True)

    for status, count in sorted(setup_status_counts.items()):
        print(f"{status}: {count}", flush=True)

    print("\nENGINE EVALUATION STATUS DISTRIBUTION", flush=True)
    print("-" * 72, flush=True)

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}", flush=True)

    print("\nDIRECTION EVALUATIONS", flush=True)
    print("-" * 72, flush=True)
    print("LONG:", direction_counts["LONG"], flush=True)
    print("SHORT:", direction_counts["SHORT"], flush=True)

    print("\nHISTORICAL STRUCTURE EVENTS", flush=True)
    print("-" * 72, flush=True)

    for event in sorted(
        event_list,
        key=lambda item: (
            item.get("available_timestamp", 0),
            item.get("index", 0),
        ),
    ):
        print(
            "Event:", event["event_type"],
            "| Direction:", event["direction"],
            "| Index:", event["index"],
            "| Timestamp:", event["timestamp"],
            "| Available:", event["available_timestamp"],
            "| Level:", event["level"],
            "| First Seen:", event["first_seen_prefix"],
            "| Last Seen:", event["last_seen_prefix"],
            "| Observations:", event["observations"],
            flush=True,
        )

    print("\nHISTORICAL SETUP DETAILS", flush=True)
    print("-" * 72, flush=True)

    for setup in sorted(
        setup_list,
        key=lambda item: (
            item.get("available_timestamp") or 0,
            item.get("setup_id") or "",
        ),
    ):
        print(
            "ID:", setup["setup_id"],
            "| Direction:", setup["direction"],
            "| Status:", setup["status"],
            "| Event:", setup["structure_event"],
            "| Available:", setup["available_timestamp"],
            "| Expiry:", setup["expiry_timestamp"],
            "| First Seen:", setup["first_seen_prefix"],
            "| Last Seen:", setup["last_seen_prefix"],
            "| Observations:", setup["observations"],
            flush=True,
        )

    print("\nALL ENGINE EVALUATIONS", flush=True)
    print("-" * 72, flush=True)

    for record in evaluation_records:
        print(
            "Prefix:", record["prefix_index"],
            "| Timestamp:", record["prefix_timestamp"],
            "| Direction:", record["direction"],
            "| Result:", record["result_status"],
            "| Setup:", record["setup_id"],
            "| Setup Status:", record["setup_status"],
            "| Event:", record["structure_event"],
            "| Event Timestamp:", record["structure_event_timestamp"],
            "| Reason:", record["reason"],
            "| Error:", record["error"],
            "| Diagnostics:",
            json.dumps(
                record["diagnostics"],
                ensure_ascii=True,
                default=str,
            ),
            flush=True,
        )

    if engine_errors:
        print("\nENGINE ERROR DETAILS", flush=True)
        print("-" * 72, flush=True)

        for error in engine_errors:
            print(
                json.dumps(
                    error,
                    ensure_ascii=True,
                    default=str,
                ),
                flush=True,
            )

    if structure_errors:
        print("\nSTRUCTURE ANALYSIS ERROR DETAILS", flush=True)
        print("-" * 72, flush=True)

        for error in structure_errors:
            print(
                json.dumps(
                    error,
                    ensure_ascii=True,
                    default=str,
                ),
                flush=True,
            )

    print("\nHISTORICAL AUDIT COMPLETED", flush=True)
    print("No trades were executed.", flush=True)
    print("Live Trading: DISABLED", flush=True)
    print("Archive Modification: NO", flush=True)
    print("=" * 72, flush=True)

    return {
        "events": event_list,
        "setups": setup_list,
        "evaluations": evaluation_records,
        "event_counts": event_type_counts,
        "setup_status_counts": setup_status_counts,
        "evaluation_status_counts": status_counts,
        "engine_errors": engine_errors,
        "structure_errors": structure_errors,
    }


def print_setup_report(result):
    print("\n4H SETUP ENGINE REPORT", flush=True)
    print("======================", flush=True)

    print("Status:", result["status"], flush=True)
    print("Allowed Direction:", result["allowed_direction"], flush=True)
    print("Reason:", result["reason"], flush=True)

    setup = result.get("setup")

    if setup is None:
        print("Setup: NONE", flush=True)
        print("Entry Confirmation: NOT AVAILABLE", flush=True)
        return None

    print("\nSETUP DETAILS", flush=True)
    print("-------------", flush=True)
    print("Setup ID:", setup["setup_id"], flush=True)
    print("Symbol:", setup["symbol"], flush=True)
    print("Direction:", setup["direction"], flush=True)
    print("Structure Event:", setup["structure_event"], flush=True)
    print("Structure Event Timestamp:", setup["structure_event_timestamp"], flush=True)
    print("Available Timestamp:", setup["available_timestamp"], flush=True)
    print("Expiry Timestamp:", setup["expiry_timestamp"], flush=True)

    print("\nENTRY ZONE", flush=True)
    print("----------", flush=True)
    print("Zone Source:", setup["entry_zone"]["source"], flush=True)
    print("Zone Low:", setup["entry_zone"]["low"], flush=True)
    print("Zone High:", setup["entry_zone"]["high"], flush=True)
    print("Reference Entry:", setup["entry_reference"], flush=True)

    print("\nRISK MANAGEMENT", flush=True)
    print("---------------", flush=True)
    print("Stop Loss:", setup["stop_loss"], flush=True)
    print("Take Profit:", setup["take_profit"], flush=True)
    print("Risk Distance:", setup["risk_distance"], flush=True)
    print("Risk Reward:", setup["risk_reward"], flush=True)

    print("\nCONFLUENCE", flush=True)
    print("----------", flush=True)
    print("Order Block:", setup["order_block"], flush=True)
    print("FVG:", setup["fvg"], flush=True)
    print("Zone Overlap:", setup["zone_overlap"], flush=True)
    print("Stop Swing:", setup["stop_swing"], flush=True)

    print("Live Trading: DISABLED", flush=True)

    return setup


def print_entry_confirmation_report(result):
    print("\nLOWER-TIMEFRAME ENTRY CONFIRMATION", flush=True)
    print("==================================", flush=True)

    if result is None:
        print("Confirmation Result: NONE", flush=True)
        print("Entry Signal: NOT GENERATED", flush=True)
        return

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
            default=str,
        ),
        flush=True,
    )

    print("\nLIVE TRADING: DISABLED", flush=True)


def run_lower_timeframe_confirmation(setup):
    print("\nCHECKING LOWER-TIMEFRAME CONFIRMATION", flush=True)
    print("=====================================", flush=True)

    if setup is None:
        print("Confirmation blocked: No valid 4H setup.", flush=True)
        return None

    if setup.get("status") != "CONFIRMED_CONFLUENCE":
        print("Confirmation blocked: Setup is not confirmed.", flush=True)
        print("Required status: CONFIRMED_CONFLUENCE", flush=True)
        print("Current status:", setup.get("status"), flush=True)
        print("Entry Signal: NOT GENERATED", flush=True)
        return None

    candles_by_timeframe = {}

    for timeframe in ("1H", "15m", "5m"):
        config = TIMEFRAME_CONFIG[timeframe]

        candles = load_structure_candles(
            config["filename"],
            config["target"],
        )

        candles_by_timeframe[timeframe] = candles

        print(
            f"{timeframe}: {len(candles)} candles loaded.",
            flush=True,
        )

    result = confirm_lower_timeframe_entry(
        setup=setup,
        candles_by_timeframe=candles_by_timeframe,
    )

    print_entry_confirmation_report(result)

    return result


def run_four_hour_setup_engine(
    market_decision,
    four_hour_candles,
):
    print("\nRUNNING 4H SETUP ENGINE", flush=True)
    print("=======================", flush=True)

    allowed_direction = market_decision["allowed_direction"]

    if allowed_direction == "WAIT":
        print("Setup generation blocked.", flush=True)
        print("Daily and 4H market direction are not aligned.", flush=True)
        return None

    result = generate_four_hour_setup(
        candles=four_hour_candles,
        allowed_direction=allowed_direction,
        setup_expiry_candles=24,
        ob_lookback=10,
        fvg_lookback=10,
        risk_reward=2.0,
    )

    setup = print_setup_report(result)

    if setup is None:
        print("\nLOWER-TIMEFRAME CONFIRMATION SKIPPED", flush=True)
        print("===================================", flush=True)
        print("No 4H setup was generated.", flush=True)
        return result

    if setup.get("status") != "CONFIRMED_CONFLUENCE":
        print("\nLOWER-TIMEFRAME CONFIRMATION SKIPPED", flush=True)
        print("===================================", flush=True)
        print("Setup is watch-only.", flush=True)
        print("OB and FVG confluence is not confirmed.", flush=True)
        print("No entry signal will be generated.", flush=True)
        return result

    run_lower_timeframe_confirmation(setup)

    return result


def main():
    print("AI CRYPTO TRADING SYSTEM", flush=True)
    print("========================", flush=True)
    print(f"Provider: {settings.DATA_PROVIDER}", flush=True)
    print(f"Market: {MARKET}", flush=True)
    print(f"Symbol: {SYMBOL}", flush=True)
    print("Timeframes: 1D, 4H, 1H, 15m, 5m", flush=True)
    print("Mode: MULTI-TIMEFRAME STRUCTURE + ENTRY CONFIRMATION", flush=True)
    print("Live Trading: DISABLED", flush=True)

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError(
            "Live trading must remain disabled."
        )

    for timeframe, config in TIMEFRAME_CONFIG.items():
        initialize_archive(timeframe, config)

    for timeframe, config in TIMEFRAME_CONFIG.items():
        update_archive(timeframe, config)

    print("\nMULTI-TIMEFRAME AUDIT", flush=True)
    print("=====================", flush=True)

    audit_results = {}

    for timeframe, config in TIMEFRAME_CONFIG.items():
        print(f"\nTIMEFRAME: {timeframe}", flush=True)

        audit_results[timeframe] = audit_archive(
            config["filename"],
            config["interval_ms"],
        )

    all_passed = all(audit_results.values())

    print("\nFINAL SYSTEM REPORT", flush=True)
    print("===================", flush=True)

    for timeframe, passed in audit_results.items():
        print(
            f"{timeframe}: "
            f"{'PASS' if passed else 'ISSUES DETECTED'}",
            flush=True,
        )

    print(
        f"OVERALL AUDIT: "
        f"{'PASS' if all_passed else 'ISSUES DETECTED'}",
        flush=True,
    )

    print("LIVE TRADING: DISABLED", flush=True)

    if not all_passed:
        raise RuntimeError(
            "One or more timeframe archives failed validation."
        )

    market_decision, four_hour_candles = (
        run_market_direction_analysis()
    )

    if RUN_HISTORICAL_CONFLUENCE_AUDIT:
        historical_audit_result = run_historical_confluence_audit(
            four_hour_candles
        )

        print("\nHISTORICAL AUDIT RETURN CHECK", flush=True)
        print("=============================", flush=True)
        print(
            "Events Returned:",
            len(historical_audit_result["events"]),
            flush=True,
        )
        print(
            "Setups Returned:",
            len(historical_audit_result["setups"]),
            flush=True,
        )
        print(
            "Evaluations Returned:",
            len(historical_audit_result["evaluations"]),
            flush=True,
        )

    if market_decision["allowed_direction"] == "WAIT":
        print("\nSETUP GENERATION BLOCKED", flush=True)
        print("========================", flush=True)
        print("Market direction is not aligned.", flush=True)
        print("No trade direction is currently allowed.", flush=True)

    else:
        print("\nSETUP DIRECTION AUTHORIZED", flush=True)
        print("==========================", flush=True)
        print(
            "Authorized Direction:",
            market_decision["allowed_direction"],
            flush=True,
        )

        run_four_hour_setup_engine(
            market_decision,
            four_hour_candles,
        )

    print("\nSYSTEM EXECUTION COMPLETED", flush=True)
    print("==========================", flush=True)
    print("Live Trading: DISABLED", flush=True)


if __name__ == "__main__":
    main()
