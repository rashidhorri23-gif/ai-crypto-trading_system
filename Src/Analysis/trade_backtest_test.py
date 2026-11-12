from pathlib import Path
import sys
import math
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"
REPORT_DIR = PROJECT_ROOT / "Reports"

TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]
ENTRY_TIMEFRAMES = ["1H", "15m", "5m"]

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1H": 3600,
    "4H": 14400,
    "1D": 86400,
}

SIGNAL_TYPES = {
    "BOS_BULLISH": 1,
    "CHOCH_BULLISH": 1,
    "BOS_BEARISH": -1,
    "CHOCH_BEARISH": -1,
}

STRUCTURE_CONFIG = {
    "INTERNAL": {
        "event_column": "internal_structure_event",
        "swing_high_column": "internal_swing_high_price",
        "swing_low_column": "internal_swing_low_price",
        "available_column": "internal_event_available_timestamp",
        "high_confirmed_column": "internal_swing_high_confirmed",
        "low_confirmed_column": "internal_swing_low_confirmed",
    },
    "EXTERNAL": {
        "event_column": "external_structure_event",
        "swing_high_column": "external_swing_high_price",
        "swing_low_column": "external_swing_low_price",
        "available_column": "external_event_available_timestamp",
        "high_confirmed_column": "external_swing_high_confirmed",
        "low_confirmed_column": "external_swing_low_confirmed",
    },
}

LEFT_BARS = 3
RIGHT_BARS = 3
BASE_HOLDING_CANDLES = 24
EXTENDED_HORIZONS = [48, 72]
RISK_REWARD = 2.0
PROFIT_LOCK_TRIGGER_R = 0.5
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
SETUP_MAX_WAIT_4H_CANDLES = 24
DIRECTION_COLUMN = "structure_trend"


# ============================================================
# DATA LOADING AND TIMESTAMP HANDLING
# ============================================================

def load_structure_module():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Src.Analysis.market_structure import detect_market_structure
    return detect_market_structure


def normalize_timestamp_value(value):
    value = float(value)
    if value > 100000000000:
        value /= 1000.0
    return int(value)


def load_archive(timeframe):
    filepath = ARCHIVE_DIR / f"BTC_USDT_SPOT_{timeframe}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Archive not found: {filepath}")

    data = pd.read_csv(filepath)
    required_columns = {
        "timestamp", "open", "high", "low", "close", "volume"
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"{timeframe}: Missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="raise")
    median_timestamp = float(data["timestamp"].median())

    if median_timestamp > 100000000000:
        data["timestamp_seconds"] = (
            data["timestamp"] / 1000.0
        ).astype("int64")
    else:
        data["timestamp_seconds"] = data["timestamp"].astype("int64")

    data = data.sort_values("timestamp_seconds").reset_index(drop=True)
    if data["timestamp_seconds"].duplicated().any():
        raise ValueError(f"{timeframe}: Duplicate timestamps found")

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not data.empty:
        if not (
            data["high"] >= data[["open", "close"]].max(axis=1)
        ).all():
            raise ValueError(f"{timeframe}: Invalid high prices")
        if not (
            data["low"] <= data[["open", "close"]].min(axis=1)
        ).all():
            raise ValueError(f"{timeframe}: Invalid low prices")
        if (data["low"] <= 0).any():
            raise ValueError(f"{timeframe}: Non-positive low prices found")
        if (data[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"{timeframe}: Non-positive OHLC prices found")
        if (data["volume"] < 0).any():
            raise ValueError(f"{timeframe}: Negative volume found")

    return data


def build_structure(data, timeframe, detect_market_structure):
    input_data = data.drop(
        columns=["timestamp_seconds"], errors="ignore"
    ).copy()

    structure = detect_market_structure(
        input_data,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
        timeframe=timeframe,
    )

    if len(structure) != len(data):
        raise ValueError(f"{timeframe}: Structure output length mismatch")

    required_columns = {DIRECTION_COLUMN}
    for config in STRUCTURE_CONFIG.values():
        required_columns.update([
            config["event_column"],
            config["swing_high_column"],
            config["swing_low_column"],
            config["available_column"],
            config["high_confirmed_column"],
            config["low_confirmed_column"],
        ])

    missing = required_columns - set(structure.columns)
    if missing:
        raise ValueError(
            f"{timeframe}: Missing structure columns: {sorted(missing)}"
        )

    return structure.reset_index(drop=True)


# ============================================================
# EVENT PARSING
# ============================================================

def parse_signal_events(event_value):
    if pd.isna(event_value):
        return []
    event_text = str(event_value).strip()
    if event_text in ("", "NONE", "NAN"):
        return []

    return [
        name.strip()
        for name in event_text.split("|")
        if name.strip() in SIGNAL_TYPES
    ]


def get_event_available_timestamp(structure, index, structure_name):
    column = STRUCTURE_CONFIG[structure_name]["available_column"]
    value = structure.iloc[index][column]
    if pd.isna(value):
        return None
    return normalize_timestamp_value(value)


def collect_structure_events(data, structure, timeframe):
    events = []
    for index in range(len(data)):
        for structure_name, config in STRUCTURE_CONFIG.items():
            event_names = parse_signal_events(
                structure.iloc[index][config["event_column"]]
            )
            if not event_names:
                continue

            available_timestamp = get_event_available_timestamp(
                structure, index, structure_name
            )
            if available_timestamp is None:
                continue

            for event_name in event_names:
                events.append({
                    "event_id": (
                        f"{timeframe}_{available_timestamp}_"
                        f"{structure_name}_{event_name}"
                    ),
                    "timeframe": timeframe,
                    "index": index,
                    "structure_type": structure_name,
                    "signal_type": event_name,
                    "direction": SIGNAL_TYPES[event_name],
                    "available_timestamp": available_timestamp,
                    "event_candle_timestamp": int(
                        data.iloc[index]["timestamp_seconds"]
                    ),
                    "event_candle_close_timestamp": (
                        int(data.iloc[index]["timestamp_seconds"])
                        + TIMEFRAME_SECONDS[timeframe]
                    ),
                })

    events.sort(key=lambda item: (
        item["available_timestamp"],
        item["index"],
        item["structure_type"],
        item["signal_type"],
    ))
    return events


# ============================================================
# DAILY DIRECTION FILTER
# ============================================================

def normalize_daily_direction(value):
    if pd.isna(value):
        return "WAIT"
    if isinstance(value, (int, float)):
        if value > 0:
            return "LONG"
        if value < 0:
            return "SHORT"
        return "WAIT"

    normalized = str(value).strip().upper()
    if normalized in ("1", "BULLISH", "BULL", "UP", "LONG"):
        return "LONG"
    if normalized in ("-1", "BEARISH", "BEAR", "DOWN", "SHORT"):
        return "SHORT"
    if normalized in ("0", "SIDEWAYS", "NEUTRAL", "NONE", "WAIT"):
        return "WAIT"
    return "WAIT"


def get_daily_bias(daily_data, daily_structure, decision_timestamp):
    daily_duration = TIMEFRAME_SECONDS["1D"]
    eligible = daily_data.index[
        (daily_data["timestamp_seconds"] + daily_duration)
        <= decision_timestamp
    ]
    if len(eligible) == 0:
        return "WAIT"
    last_index = int(eligible[-1])
    return normalize_daily_direction(
        daily_structure.iloc[last_index][DIRECTION_COLUMN]
    )


# ============================================================
# TRADE COSTS AND EXCURSIONS
# ============================================================

def calculate_trade_costs(direction, entry_price, exit_price):
    if direction == 1:
        entry_fill = entry_price * (1 + SLIPPAGE_RATE)
        exit_fill = exit_price * (1 - SLIPPAGE_RATE)
        gross_pnl = exit_price - entry_price
        net_pnl = (
            exit_fill - entry_fill
            - FEE_RATE * (entry_fill + exit_fill)
        )
    else:
        entry_fill = entry_price * (1 - SLIPPAGE_RATE)
        exit_fill = exit_price * (1 + SLIPPAGE_RATE)
        gross_pnl = entry_price - exit_price
        net_pnl = (
            entry_fill - exit_fill
            - FEE_RATE * (entry_fill + exit_fill)
        )
    return gross_pnl, net_pnl, entry_fill


def calculate_excursions(
    data, entry_index, exit_index, direction, entry_price, risk_per_unit
):
    candles = data.iloc[entry_index:exit_index + 1]
    highest = float(candles["high"].max())
    lowest = float(candles["low"].min())

    if direction == 1:
        favorable = max(0.0, highest - entry_price)
        adverse = max(0.0, entry_price - lowest)
    else:
        favorable = max(0.0, entry_price - lowest)
        adverse = max(0.0, highest - entry_price)

    return {
        "mfe_R": favorable / risk_per_unit,
        "mae_R": adverse / risk_per_unit,
        "mfe_pct": favorable / entry_price * 100,
        "mae_pct": adverse / entry_price * 100,
    }


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(
    data,
    signal_index,
    direction,
    stop_price,
    max_holding_candles,
    use_profit_lock=False,
):
    entry_index = signal_index + 1
    if entry_index + max_holding_candles > len(data):
        return None, "INSUFFICIENT_FUTURE_DATA"

    entry_price = float(data.iloc[entry_index]["open"])
    initial_stop = float(stop_price)

    if (
        not math.isfinite(entry_price)
        or not math.isfinite(initial_stop)
        or entry_price <= 0
        or initial_stop <= 0
    ):
        return None, "INVALID_PRICE"

    if direction == 1 and initial_stop >= entry_price:
        return None, "INVALID_LONG_STOP"
    if direction == -1 and initial_stop <= entry_price:
        return None, "INVALID_SHORT_STOP"

    risk = abs(entry_price - initial_stop)
    if risk <= 0:
        return None, "ZERO_RISK"

    if direction == 1:
        target = entry_price + RISK_REWARD * risk
        lock_trigger = entry_price + PROFIT_LOCK_TRIGGER_R * risk
    else:
        target = entry_price - RISK_REWARD * risk
        lock_trigger = entry_price - PROFIT_LOCK_TRIGGER_R * risk

    active_stop = initial_stop
    lock_activated = False
    activate_lock_next = False
    exit_price = None
    exit_reason = None
    exit_index = None
    end_index = entry_index + max_holding_candles

    for current_index in range(entry_index, end_index):
        candle = data.iloc[current_index]
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])

        if activate_lock_next:
            active_stop = lock_trigger
            lock_activated = True
            activate_lock_next = False

        if direction == 1:
            if open_price <= active_stop:
                exit_price = open_price
                exit_reason = (
                    "PROFIT_LOCK_GAP" if lock_activated else "STOP_GAP"
                )
            elif open_price >= target:
                exit_price = open_price
                exit_reason = "TARGET_GAP"
            else:
                stop_hit = low_price <= active_stop
                target_hit = high_price >= target
                if stop_hit:
                    exit_price = active_stop
                    if lock_activated:
                        exit_reason = "PROFIT_LOCK"
                    elif target_hit:
                        exit_reason = "STOP_BOTH_HIT"
                    else:
                        exit_reason = "STOP"
                elif target_hit:
                    exit_price = target
                    exit_reason = "TARGET"
        else:
            if open_price >= active_stop:
                exit_price = open_price
                exit_reason = (
                    "PROFIT_LOCK_GAP" if lock_activated else "STOP_GAP"
                )
            elif open_price <= target:
                exit_price = open_price
                exit_reason = "TARGET_GAP"
            else:
                stop_hit = high_price >= active_stop
                target_hit = low_price <= target
                if stop_hit:
                    exit_price = active_stop
                    if lock_activated:
                        exit_reason = "PROFIT_LOCK"
                    elif target_hit:
                        exit_reason = "STOP_BOTH_HIT"
                    else:
                        exit_reason = "STOP"
                elif target_hit:
                    exit_price = target
                    exit_reason = "TARGET"

        if exit_price is not None:
            exit_index = current_index
            break

        if use_profit_lock and not lock_activated:
            if direction == 1 and high_price >= lock_trigger:
                activate_lock_next = True
            elif direction == -1 and low_price <= lock_trigger:
                activate_lock_next = True

    if exit_price is None:
        exit_index = end_index - 1
        exit_price = float(data.iloc[exit_index]["close"])
        exit_reason = "TIME_EXIT"

    gross_pnl, net_pnl, entry_fill = calculate_trade_costs(
        direction, entry_price, exit_price
    )
    gross_r = gross_pnl / risk
    net_r = net_pnl / risk
    net_return_pct = net_pnl / entry_fill * 100

    excursions = calculate_excursions(
        data, entry_index, exit_index, direction, entry_price, risk
    )

    trade = {
        "signal_timestamp": int(
            data.iloc[signal_index]["timestamp_seconds"]
        ),
        "entry_timestamp": int(
            data.iloc[entry_index]["timestamp_seconds"]
        ),
        "exit_timestamp": int(
            data.iloc[exit_index]["timestamp_seconds"]
        ),
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry_price": entry_price,
        "stop_price": initial_stop,
        "target_price": target,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_R": gross_r,
        "net_R": net_r,
        "net_return_pct": net_return_pct,
        "risk_per_unit": risk,
        "holding_candles": exit_index - entry_index + 1,
        "profit_lock_activated": lock_activated,
        "profit_lock_price": lock_trigger if lock_activated else None,
        "mfe_R": excursions["mfe_R"],
        "mae_R": excursions["mae_R"],
        "mfe_pct": excursions["mfe_pct"],
        "mae_pct": excursions["mae_pct"],
    }
    return trade, None


# ============================================================
# 4H SETUP GENERATION AND STOP AUDIT
# ============================================================

def get_latest_confirmed_swings(structure, data):
    result = {}

    for name, config in STRUCTURE_CONFIG.items():
        high_confirmed = structure[config["high_confirmed_column"]].astype(bool)
        low_confirmed = structure[config["low_confirmed_column"]].astype(bool)

        high_confirmation_time = pd.Series(
            pd.NA, index=structure.index, dtype="Int64"
        )
        low_confirmation_time = pd.Series(
            pd.NA, index=structure.index, dtype="Int64"
        )

        high_confirmation_time.loc[high_confirmed] = (
            data.loc[high_confirmed, "timestamp_seconds"].astype("int64")
        )
        low_confirmation_time.loc[low_confirmed] = (
            data.loc[low_confirmed, "timestamp_seconds"].astype("int64")
        )

        result[name] = {
            "high": structure[config["swing_high_column"]].ffill(),
            "low": structure[config["swing_low_column"]].ffill(),
            "high_confirmation_timestamp": high_confirmation_time.ffill(),
            "low_confirmation_timestamp": low_confirmation_time.ffill(),
        }

    return result


def create_4h_setups(
    four_hour_data, four_hour_structure, daily_data, daily_structure
):
    setups = []
    rejected = {}
    latest_swings = get_latest_confirmed_swings(
        four_hour_structure, four_hour_data
    )
    events = collect_structure_events(
        four_hour_data, four_hour_structure, "4H"
    )

    for event in events:
        index = event["index"]
        available = event["available_timestamp"]
        bias = get_daily_bias(daily_data, daily_structure, available)

        if bias == "WAIT":
            rejected["DAILY_BIAS_WAIT"] = (
                rejected.get("DAILY_BIAS_WAIT", 0) + 1
            )
            continue

        expected_direction = 1 if bias == "LONG" else -1
        if event["direction"] != expected_direction:
            rejected["DAILY_DIRECTION_CONFLICT"] = (
                rejected.get("DAILY_DIRECTION_CONFLICT", 0) + 1
            )
            continue

        name = event["structure_type"]
        if event["direction"] == 1:
            stop = latest_swings[name]["low"].iloc[index]
            stop_confirmation = latest_swings[name][
                "low_confirmation_timestamp"
            ].iloc[index]
        else:
            stop = latest_swings[name]["high"].iloc[index]
            stop_confirmation = latest_swings[name][
                "high_confirmation_timestamp"
            ].iloc[index]

        if pd.isna(stop) or pd.isna(stop_confirmation):
            rejected["NO_CONFIRMED_4H_OPPOSITE_SWING"] = (
                rejected.get("NO_CONFIRMED_4H_OPPOSITE_SWING", 0) + 1
            )
            continue

        stop_confirmation = int(stop_confirmation)
        if stop_confirmation > available:
            rejected["STOP_SWING_NOT_AVAILABLE_AT_SETUP"] = (
                rejected.get("STOP_SWING_NOT_AVAILABLE_AT_SETUP", 0) + 1
            )
            continue

        setups.append({
            "setup_id": (
                f"4H_{available}_{name}_{event['signal_type']}"
            ),
            "setup_index": index,
            "setup_timestamp": int(
                four_hour_data.iloc[index]["timestamp_seconds"]
            ),
            "available_timestamp": available,
            "direction": event["direction"],
            "daily_bias": bias,
            "structure_type": name,
            "signal_type": event["signal_type"],
            "stop_price": float(stop),
            "stop_swing_confirmation_timestamp": stop_confirmation,
            "stop_swing_confirmation_lag_seconds": (
                available - stop_confirmation
            ),
        })

    setups.sort(key=lambda item: (
        item["available_timestamp"], item["setup_id"]
    ))
    return setups, rejected, events


# ============================================================
# LOWER TIMEFRAME ENTRY CONFIRMATION
# ============================================================

def collect_lower_timeframe_events(archives, structures):
    events = []
    for timeframe in ENTRY_TIMEFRAMES:
        events.extend(collect_structure_events(
            archives[timeframe], structures[timeframe], timeframe
        ))

    priority = {"1H": 0, "15m": 1, "5m": 2}
    events.sort(key=lambda item: (
        item["available_timestamp"],
        priority[item["timeframe"]],
        item["index"],
        item["event_id"],
    ))
    return events


def find_entry_confirmation(
    setup, lower_events, expiry_timestamp, consumed_event_ids
):
    candidates = [
        event for event in lower_events
        if (
            event["available_timestamp"] >= setup["available_timestamp"]
            and event["available_timestamp"] <= expiry_timestamp
            and event["direction"] == setup["direction"]
            and event["event_id"] not in consumed_event_ids
        )
    ]
    if not candidates:
        return None

    priority = {"1H": 0, "15m": 1, "5m": 2}
    candidates.sort(key=lambda item: (
        item["available_timestamp"],
        priority[item["timeframe"]],
        item["index"],
        item["event_id"],
    ))
    return candidates[0]


def build_entry_candidate_audit(
    setup, lower_events, expiry_timestamp, consumed_event_ids,
    selected_event_id=None
):
    rows = []
    priority = {"1H": 0, "15m": 1, "5m": 2}

    candidates = [
        event for event in lower_events
        if (
            setup["available_timestamp"]
            <= event["available_timestamp"]
            <= expiry_timestamp
        )
    ]
    candidates.sort(key=lambda item: (
        item["available_timestamp"],
        priority[item["timeframe"]],
        item["index"],
        item["event_id"],
    ))

    first_time = (
        candidates[0]["available_timestamp"] if candidates else None
    )
    same_time_count = (
        sum(e["available_timestamp"] == first_time for e in candidates)
        if first_time is not None else 0
    )

    for event in candidates:
        if event["direction"] != setup["direction"]:
            reason = "DIRECTION_MISMATCH"
        elif event["event_id"] in consumed_event_ids:
            reason = "ALREADY_CONSUMED_BY_EARLIER_SETUP"
        elif event["event_id"] == selected_event_id:
            reason = "SELECTED"
        else:
            reason = "NOT_SELECTED_EARLIER_ELIGIBLE_EVENT"

        rows.append({
            "setup_id": setup["setup_id"],
            "setup_direction": (
                "LONG" if setup["direction"] == 1 else "SHORT"
            ),
            "setup_available_timestamp": setup["available_timestamp"],
            "setup_expiry_timestamp": expiry_timestamp,
            "event_id": event["event_id"],
            "entry_timeframe": event["timeframe"],
            "entry_structure_type": event["structure_type"],
            "entry_signal_type": event["signal_type"],
            "event_direction": (
                "LONG" if event["direction"] == 1 else "SHORT"
            ),
            "event_candle_timestamp": event["event_candle_timestamp"],
            "confirmation_timestamp": event["available_timestamp"],
            "confirmation_lag_from_setup_seconds": (
                event["available_timestamp"]
                - setup["available_timestamp"]
            ),
            "timeframe_priority": priority[event["timeframe"]],
            "same_timestamp_candidate_count": same_time_count,
            "same_timestamp_tie": (
                same_time_count > 1
                and event["available_timestamp"] == first_time
            ),
            "selected": event["event_id"] == selected_event_id,
            "audit_reason": reason,
        })

    return rows


# ============================================================
# SETUP INVALIDATION
# ============================================================

def is_setup_invalidated(
    setup, entry_event, four_hour_data, four_hour_events
):
    setup_time = setup["available_timestamp"]
    entry_time = entry_event["available_timestamp"]
    direction = setup["direction"]
    stop = setup["stop_price"]

    for index in range(len(four_hour_data)):
        candle_open_time = int(
            four_hour_data.iloc[index]["timestamp_seconds"]
        )
        candle_close_time = (
            candle_open_time + TIMEFRAME_SECONDS["4H"]
        )

        if candle_close_time <= setup_time:
            continue
        if candle_close_time > entry_time:
            continue

        candle = four_hour_data.iloc[index]
        if direction == 1 and float(candle["low"]) <= stop:
            return True, "STOP_TOUCHED_BEFORE_ENTRY"
        if direction == -1 and float(candle["high"]) >= stop:
            return True, "STOP_TOUCHED_BEFORE_ENTRY"

    for event in four_hour_events:
        if not (
            setup_time < event["available_timestamp"] <= entry_time
        ):
            continue
        if event["direction"] != direction:
            return True, "OPPOSITE_4H_STRUCTURE_EVENT"

    return False, None


# ============================================================
# HOLDING PERIOD AND SUMMARY REPORTS
# ============================================================

def get_holding_candles(entry_timeframe, four_hour_horizon):
    holding_seconds = four_hour_horizon * TIMEFRAME_SECONDS["4H"]
    entry_seconds = TIMEFRAME_SECONDS[entry_timeframe]
    return int(holding_seconds // entry_seconds)


def calculate_summary(trades):
    if not trades:
        return {
            "trades": 0, "win_rate_pct": 0, "avg_net_R": 0,
            "median_net_R": 0, "profit_factor": 0,
            "avg_mfe_R": 0, "avg_mae_R": 0,
            "target_exits": 0, "stop_exits": 0,
            "profit_lock_exits": 0, "time_exits": 0,
        }

    frame = pd.DataFrame(trades)
    net_r = frame["net_R"]
    winners = net_r[net_r > 0]
    losers = net_r[net_r < 0]

    if len(losers):
        pf = winners.sum() / abs(losers.sum())
    else:
        pf = float("inf") if len(winners) else 0

    return {
        "trades": len(frame),
        "win_rate_pct": round((net_r > 0).mean() * 100, 2),
        "avg_net_R": round(net_r.mean(), 4),
        "median_net_R": round(net_r.median(), 4),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "INF",
        "avg_mfe_R": round(frame["mfe_R"].mean(), 4),
        "avg_mae_R": round(frame["mae_R"].mean(), 4),
        "target_exits": int(frame["exit_reason"].str.startswith("TARGET").sum()),
        "stop_exits": int(frame["exit_reason"].str.startswith("STOP").sum()),
        "profit_lock_exits": int(
            frame["exit_reason"].str.startswith("PROFIT_LOCK").sum()
        ),
        "time_exits": int((frame["exit_reason"] == "TIME_EXIT").sum()),
    }


def print_grouped_summary(frame, group_columns, heading):
    print(f"\n{heading}")
    if frame.empty:
        print("No trades available.")
        return

    for key, group in frame.groupby(group_columns):
        if not isinstance(key, tuple):
            key = (key,)
        label = " | ".join(str(value) for value in key)
        print(f"{label}: {calculate_summary(group.to_dict('records'))}")


def print_extension_report(extensions):
    print("\nBASELINE EXTENDED HOLDING ANALYSIS")
    if not extensions:
        print("No eligible Time Exit trades for extension analysis.")
        return

    frame = pd.DataFrame(extensions)
    for horizon in EXTENDED_HORIZONS:
        subset = frame[
            (frame["horizon_4h"] == horizon)
            & (frame["status"] == "AVAILABLE")
        ]
        unavailable = frame[
            (frame["horizon_4h"] == horizon)
            & (frame["status"] != "AVAILABLE")
        ]

        print(f"\nHORIZON: {horizon} equivalent 4H candles")
        print(f"Available trades: {len(subset)}")
        print(f"Unavailable trades: {len(unavailable)}")
        if subset.empty:
            continue

        print("Summary:", calculate_summary(subset.to_dict("records")))
        print("Exit reasons:", subset["exit_reason"].value_counts().to_dict())
        print("By entry timeframe:")
        for timeframe, group in subset.groupby("entry_timeframe"):
            print(
                timeframe,
                calculate_summary(group.to_dict("records"))
            )


def timestamp_to_utc_text(value):
    if value is None or pd.isna(value):
        return ""
    return pd.to_datetime(
        int(value), unit="s", utc=True
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def save_diagnostic_reports(
    trade_diagnostics, candidate_audit, setup_status_audit,
    baseline_trades, managed_trades, extensions
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    trade_frame = pd.DataFrame(trade_diagnostics)
    candidate_frame = pd.DataFrame(candidate_audit)
    setup_frame = pd.DataFrame(setup_status_audit)

    performance_rows = []
    for label, trades in (
        ("BASELINE", baseline_trades),
        ("PROFIT_LOCK", managed_trades),
    ):
        frame = pd.DataFrame(trades)
        if frame.empty:
            continue
        for group_column in (
            "entry_timeframe",
            "daily_bias",
            "setup_signal_type",
            "entry_signal_type",
        ):
            if group_column not in frame.columns:
                continue
            for group_value, group in frame.groupby(group_column):
                performance_rows.append({
                    "mode": label,
                    "group_by": group_column,
                    "group_value": group_value,
                    **calculate_summary(group.to_dict("records")),
                })

    files = {
        "trade_diagnostics": (
            REPORT_DIR / "BTC_Trade_Diagnostics.csv", trade_frame
        ),
        "entry_candidate_audit": (
            REPORT_DIR / "BTC_Entry_Confirmation_Audit.csv",
            candidate_frame,
        ),
        "setup_status_audit": (
            REPORT_DIR / "BTC_Setup_Status_Audit.csv", setup_frame
        ),
        "performance_breakdown": (
            REPORT_DIR / "BTC_Performance_Breakdown.csv",
            pd.DataFrame(performance_rows),
        ),
        "extended_holding": (
            REPORT_DIR / "BTC_Extended_Holding_Audit.csv",
            pd.DataFrame(extensions),
        ),
    }

    for _, (path, frame) in files.items():
        frame.to_csv(path, index=False)

    print("\nDIAGNOSTIC REPORT FILES")
    for name, (path, frame) in files.items():
        print(f"{name}: {path} ({len(frame)} rows)")


# ============================================================
# HIERARCHICAL BACKTEST ENGINE
# ============================================================

def analyze_hierarchical_strategy(detect_market_structure):
    print("\n" + "=" * 72)
    print("HIERARCHICAL MULTI-TIMEFRAME BACKTEST")
    print("=" * 72)
    print("Direction filter: 1D")
    print("Setup timeframe: 4H")
    print("Entry timeframes: 1H, 15m, 5m")
    print("Entry confirmation: First aligned BOS/CHOCH")
    print("Stop source: Confirmed 4H opposite swing")
    print(f"Base holding period: {BASE_HOLDING_CANDLES} equivalent 4H candles")
    print(
        f"Setup expiry: {SETUP_MAX_WAIT_4H_CANDLES} equivalent 4H candles"
    )
    print("Trading execution: DISABLED")
    print("Archive modification: NO")

    archives = {}
    structures = {}
    for timeframe in TIMEFRAMES:
        archives[timeframe] = load_archive(timeframe)
        structures[timeframe] = build_structure(
            archives[timeframe], timeframe, detect_market_structure
        )
        print(f"{timeframe}: {len(archives[timeframe])} candles loaded")

    setups, rejected_setups, four_hour_events = create_4h_setups(
        archives["4H"], structures["4H"],
        archives["1D"], structures["1D"],
    )

    print(f"\nDaily-filtered 4H setups: {len(setups)}")
    print(f"Rejected 4H events: {sum(rejected_setups.values())}")
    for reason, count in sorted(rejected_setups.items()):
        print(f"  {reason}: {count}")

    lower_events = collect_lower_timeframe_events(archives, structures)
    print(f"\nLower-timeframe structure events: {len(lower_events)}")
    for timeframe in ENTRY_TIMEFRAMES:
        count = sum(e["timeframe"] == timeframe for e in lower_events)
        print(f"  {timeframe}: {count} events")

    # The data horizon is the close time of the last available candle
    # in each lower timeframe. The maximum is used to classify setup expiry.
    latest_lower_data_timestamp = max(
        int(archives[tf].iloc[-1]["timestamp_seconds"])
        + TIMEFRAME_SECONDS[tf]
        for tf in ENTRY_TIMEFRAMES
        if not archives[tf].empty
    )

    baseline_trades = []
    managed_trades = []
    extensions = []
    trade_diagnostics = []
    candidate_audit = []
    setup_status_audit = []
    consumed_event_ids = set()

    statistics = {
        "valid_4h_setups": len(setups),
        "entry_confirmed": 0,
        "pending": 0,
        "expired": 0,
        "invalidated_before_entry": 0,
        "insufficient_future_data": 0,
        "invalid_trades": 0,
        "entry_timeframes": {},
        "invalidation_reasons": {},
    }

    timeframe_priority = {"1H": 0, "15m": 1, "5m": 2}

    for setup in setups:
        expiry = (
            setup["available_timestamp"]
            + SETUP_MAX_WAIT_4H_CANDLES * TIMEFRAME_SECONDS["4H"]
        )

        # Audit all lower-TF events in the setup window, including
        # opposite-direction events and events consumed by earlier setups.
        window_events = [
            event for event in lower_events
            if setup["available_timestamp"]
            <= event["available_timestamp"] <= expiry
        ]
        available_candidates = [
            event for event in window_events
            if (
                event["direction"] == setup["direction"]
                and event["event_id"] not in consumed_event_ids
            )
        ]
        available_candidates.sort(key=lambda event: (
            event["available_timestamp"],
            timeframe_priority[event["timeframe"]],
            event["index"],
            event["event_id"],
        ))

        selected = available_candidates[0] if available_candidates else None
        selected_id = selected["event_id"] if selected else None

        candidate_audit.extend(build_entry_candidate_audit(
            setup, lower_events, expiry, consumed_event_ids, selected_id
        ))

        if selected is None:
            if expiry > latest_lower_data_timestamp:
                status = "PENDING"
                statistics["pending"] += 1
                status_reason = "SETUP_EXPIRY_AFTER_AVAILABLE_DATA"
            else:
                status = "EXPIRED"
                statistics["expired"] += 1
                status_reason = (
                    "NO_UNCONSUMED_ALIGNED_ENTRY_EVENT_BEFORE_EXPIRY"
                )

            setup_status_audit.append({
                **setup,
                "setup_expiry_timestamp": expiry,
                "status": status,
                "status_reason": status_reason,
                "selected_entry_event_id": "",
                "selected_entry_timeframe": "",
                "entry_confirmation_timestamp": "",
                "latest_lower_data_timestamp": latest_lower_data_timestamp,
            })
            continue

        # Consume the event at assignment time, not after successful
        # simulation. One event can never be assigned to another setup.
        consumed_event_ids.add(selected_id)

        invalidated, invalidation_reason = is_setup_invalidated(
            setup, selected, archives["4H"], four_hour_events
        )

        entry_tf = selected["timeframe"]
        entry_data = archives[entry_tf]
        entry_index = selected["index"]
        next_index = entry_index + 1

        if next_index >= len(entry_data):
            base_trade = None
            reason = "INSUFFICIENT_FUTURE_DATA"
        else:
            next_open_timestamp = int(
                entry_data.iloc[next_index]["timestamp_seconds"]
            )
            confirmation_timestamp = selected["available_timestamp"]

            # A candle cannot be used as an entry if it opens before
            # the structure event becomes available.
            if next_open_timestamp < confirmation_timestamp:
                base_trade = None
                reason = "ENTRY_CANDLE_OPENS_BEFORE_CONFIRMATION"
            else:
                holding = get_holding_candles(
                    entry_tf, BASE_HOLDING_CANDLES
                )
                base_trade, reason = simulate_trade(
                    entry_data, entry_index, setup["direction"],
                    setup["stop_price"], holding, use_profit_lock=False,
                )

        if invalidated:
            status = "INVALIDATED"
            status_reason = invalidation_reason
            statistics["invalidated_before_entry"] += 1
            statistics["invalidation_reasons"][invalidation_reason] = (
                statistics["invalidation_reasons"].get(
                    invalidation_reason, 0
                ) + 1
            )
            base_trade = None
            reason = invalidation_reason
        elif reason:
            status = (
                "INSUFFICIENT_FUTURE_DATA"
                if reason == "INSUFFICIENT_FUTURE_DATA"
                else "INVALID_TRADE"
            )
            status_reason = reason
            if status == "INSUFFICIENT_FUTURE_DATA":
                statistics["insufficient_future_data"] += 1
            else:
                statistics["invalid_trades"] += 1
        else:
            status = "ENTRY_CONFIRMED"
            status_reason = "TRADE_SIMULATED"

        metadata = {
            "setup_id": setup["setup_id"],
            "setup_timeframe": "4H",
            "setup_timestamp": setup["setup_timestamp"],
            "setup_available_timestamp": setup["available_timestamp"],
            "setup_type": setup["structure_type"],
            "setup_signal_type": setup["signal_type"],
            "daily_bias": setup["daily_bias"],
            "setup_direction": (
                "LONG" if setup["direction"] == 1 else "SHORT"
            ),
            "stop_swing_confirmation_timestamp": (
                setup["stop_swing_confirmation_timestamp"]
            ),
            "stop_swing_confirmation_lag_seconds": (
                setup["stop_swing_confirmation_lag_seconds"]
            ),
            "entry_event_id": selected["event_id"],
            "entry_timeframe": entry_tf,
            "entry_structure_type": selected["structure_type"],
            "entry_signal_type": selected["signal_type"],
            "entry_event_candle_timestamp": (
                selected["event_candle_timestamp"]
            ),
            "entry_confirmation_timestamp": (
                selected["available_timestamp"]
            ),
            "confirmation_lag_seconds": (
                selected["available_timestamp"]
                - setup["available_timestamp"]
            ),
            "entry_open_timestamp": (
                int(entry_data.iloc[next_index]["timestamp_seconds"])
                if next_index < len(entry_data) else None
            ),
            "confirmation_to_entry_seconds": (
                int(entry_data.iloc[next_index]["timestamp_seconds"])
                - selected["available_timestamp"]
                if next_index < len(entry_data) else None
            ),
            "setup_expiry_timestamp": expiry,
            "holding_period_4h_equivalent": BASE_HOLDING_CANDLES,
            "event_consumed_at_assignment": True,
        }

        setup_status_audit.append({
            **setup,
            "setup_expiry_timestamp": expiry,
            "status": status,
            "status_reason": status_reason,
            "selected_entry_event_id": selected_id,
            "selected_entry_timeframe": entry_tf,
            "entry_confirmation_timestamp": selected["available_timestamp"],
            "latest_lower_data_timestamp": latest_lower_data_timestamp,
        })

        if status != "ENTRY_CONFIRMED":
            trade_diagnostics.append({
                **metadata,
                "trade_status": status,
                "status_reason": status_reason,
                "baseline_net_R": None,
                "managed_net_R": None,
            })
            continue

        managed_trade, managed_reason = simulate_trade(
            entry_data, entry_index, setup["direction"],
            setup["stop_price"],
            get_holding_candles(entry_tf, BASE_HOLDING_CANDLES),
            use_profit_lock=True,
        )

        if managed_reason:
            statistics["invalid_trades"] += 1
            setup_status_audit[-1]["status"] = "INVALID_TRADE"
            setup_status_audit[-1]["status_reason"] = managed_reason
            trade_diagnostics.append({
                **metadata,
                "trade_status": "INVALID_TRADE",
                "status_reason": managed_reason,
                "baseline_net_R": base_trade["net_R"],
                "managed_net_R": None,
            })
            continue

        base_trade.update(metadata)
        managed_trade.update(metadata)
        baseline_trades.append(base_trade)
        managed_trades.append(managed_trade)

        statistics["entry_confirmed"] += 1
        statistics["entry_timeframes"][entry_tf] = (
            statistics["entry_timeframes"].get(entry_tf, 0) + 1
        )

        trade_diagnostics.append({
            **metadata,
            "trade_status": "CLOSED",
            "status_reason": "TRADE_SIMULATED",
            "entry_price": base_trade["entry_price"],
            "stop_price": base_trade["stop_price"],
            "target_price": base_trade["target_price"],
            "baseline_exit_timestamp": base_trade["exit_timestamp"],
            "baseline_exit_price": base_trade["exit_price"],
            "baseline_exit_reason": base_trade["exit_reason"],
            "baseline_gross_R": base_trade["gross_R"],
            "baseline_net_R": base_trade["net_R"],
            "baseline_mfe_R": base_trade["mfe_R"],
            "baseline_mae_R": base_trade["mae_R"],
            "managed_exit_timestamp": managed_trade["exit_timestamp"],
            "managed_exit_price": managed_trade["exit_price"],
            "managed_exit_reason": managed_trade["exit_reason"],
            "managed_gross_R": managed_trade["gross_R"],
            "managed_net_R": managed_trade["net_R"],
            "managed_mfe_R": managed_trade["mfe_R"],
            "managed_mae_R": managed_trade["mae_R"],
        })

        if base_trade["exit_reason"] != "TIME_EXIT":
            continue

        for horizon in EXTENDED_HORIZONS:
            extended_holding = get_holding_candles(entry_tf, horizon)
            extended_trade, extension_reason = simulate_trade(
                entry_data, entry_index, setup["direction"],
                setup["stop_price"], extended_holding,
                use_profit_lock=False,
            )
            if extension_reason:
                extensions.append({
                    **metadata,
                    "entry_timeframe": entry_tf,
                    "horizon_4h": horizon,
                    "status": extension_reason,
                })
            else:
                extended_trade.update(metadata)
                extended_trade["horizon_4h"] = horizon
                extended_trade["status"] = "AVAILABLE"
                extensions.append(extended_trade)

    # --------------------------------------------------------
    # Final reports
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HIERARCHICAL BACKTEST RESULTS")
    print("=" * 72)
    print(f"Valid 4H setups: {statistics['valid_4h_setups']}")
    print(f"Entry-confirmed trades: {statistics['entry_confirmed']}")
    print(f"Pending setups: {statistics['pending']}")
    print(f"Expired setups: {statistics['expired']}")
    print(
        f"Invalidated before entry: "
        f"{statistics['invalidated_before_entry']}"
    )
    print(
        f"Insufficient future data: "
        f"{statistics['insufficient_future_data']}"
    )
    print(f"Invalid trades: {statistics['invalid_trades']}")

    print("\nEntries by timeframe:")
    for timeframe in ENTRY_TIMEFRAMES:
        print(
            f"  {timeframe}: "
            f"{statistics['entry_timeframes'].get(timeframe, 0)}"
        )

    print("\nInvalidation reasons:")
    for reason, count in sorted(statistics["invalidation_reasons"].items()):
        print(f"  {reason}: {count}")

    print("\nBASELINE SUMMARY")
    print(calculate_summary(baseline_trades))
    print("\nPROFIT LOCK SUMMARY")
    print(calculate_summary(managed_trades))

    if baseline_trades:
        baseline_frame = pd.DataFrame(baseline_trades)
        managed_frame = pd.DataFrame(managed_trades)

        print_grouped_summary(
            baseline_frame, ["entry_timeframe"],
            "BASELINE BY ENTRY TIMEFRAME",
        )
        print_grouped_summary(
            managed_frame, ["entry_timeframe"],
            "PROFIT LOCK BY ENTRY TIMEFRAME",
        )
        print_grouped_summary(
            managed_frame, ["daily_bias", "setup_signal_type"],
            "PROFIT LOCK BY DAILY BIAS AND 4H SETUP",
        )
        print("\nPROFIT LOCK EXIT REASONS")
        print(managed_frame["exit_reason"].value_counts().to_dict())

    print_extension_report(extensions)

    save_diagnostic_reports(
        trade_diagnostics,
        candidate_audit,
        setup_status_audit,
        baseline_trades,
        managed_trades,
        extensions,
    )

    print("\n" + "=" * 72)
    print("DATA AND METHODOLOGY NOTES")
    print("=" * 72)
    print("Daily direction uses only completed daily candles.")
    print("Only 4H BOS/CHOCH events create trade setups.")
    print("Lower timeframes are used only for entry confirmation.")
    print("Each 4H setup can produce at most one trade.")
    print("Each lower-timeframe confirmation event is consumed at assignment.")
    print("Stops use confirmed 4H opposite swings available at setup time.")
    print("Holding periods are measured in equivalent 4H candles.")
    print("Same-candle stop and target conflicts use stop-first logic.")
    print("Profit Lock activates on the candle after its trigger.")
    print("No live orders are submitted.")
    print("\nHIERARCHICAL BACKTEST COMPLETED")

    return {
        "setups": setups,
        "baseline_trades": baseline_trades,
        "managed_trades": managed_trades,
        "extensions": extensions,
        "statistics": statistics,
        "trade_diagnostics": trade_diagnostics,
        "entry_candidate_audit": candidate_audit,
        "setup_status_audit": setup_status_audit,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("BTC HIERARCHICAL MARKET STRUCTURE BACKTEST")
    print("=" * 72)
    print("Direction timeframe: 1D")
    print("Setup timeframe: 4H")
    print("Entry timeframes: 1H, 15m, 5m")
    print(f"Risk/Reward: 1:{RISK_REWARD}")
    print(
        f"Base holding period: {BASE_HOLDING_CANDLES} equivalent 4H candles"
    )
    print(
        f"Setup expiry: {SETUP_MAX_WAIT_4H_CANDLES} equivalent 4H candles"
    )
    print(f"Profit lock trigger: +{PROFIT_LOCK_TRIGGER_R}R")
    print(f"Extended horizons: {EXTENDED_HORIZONS}")
    print(f"Fee per side: {FEE_RATE * 100:.3f}%")
    print(f"Slippage per side: {SLIPPAGE_RATE * 100:.3f}%")
    print("Entry price: Next lower-timeframe candle Open")
    print("Same-candle SL/TP: Stop Loss First")
    print("Profit lock activation: Next candle after trigger")
    print("Trading execution: DISABLED")
    print("Archive modification: NO")

    detect_market_structure = load_structure_module()
    analyze_hierarchical_strategy(detect_market_structure)


if __name__ == "__main__":
    main()
