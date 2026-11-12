
from pathlib import Path
import sys
import math
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

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
    },
    "EXTERNAL": {
        "event_column": "external_structure_event",
        "swing_high_column": "external_swing_high_price",
        "swing_low_column": "external_swing_low_price",
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

SETUP_TIMEFRAME = "4H"
DIRECTION_TIMEFRAME = "1D"

SETUP_MAX_WAIT_4H_CANDLES = 24

DAILY_TREND_COLUMNS = [
    "structure_trend",
    "external_structure_trend",
    "trend",
]


def load_structure_module():
    sys.path.insert(0, str(PROJECT_ROOT))
    from Src.Analysis.market_structure import detect_market_structure
    return detect_market_structure


def normalize_timestamps(data):
    result = data.copy()

    result["timestamp"] = pd.to_numeric(
        result["timestamp"],
        errors="raise",
    )

    median_timestamp = float(result["timestamp"].median())

    if median_timestamp > 100000000000:
        result["timestamp_seconds"] = (
            result["timestamp"] / 1000.0
        )
    else:
        result["timestamp_seconds"] = result["timestamp"]

    result["timestamp_seconds"] = (
        result["timestamp_seconds"].astype("int64")
    )

    return result


def load_archive(timeframe):
    filename = f"BTC_USDT_SPOT_{timeframe}.csv"
    filepath = ARCHIVE_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Archive not found: {filepath}"
        )

    data = pd.read_csv(filepath)

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{timeframe}: Missing columns: {sorted(missing)}"
        )

    data = normalize_timestamps(data)

    data = data.sort_values(
        "timestamp_seconds"
    ).reset_index(drop=True)

    if data["timestamp_seconds"].duplicated().any():
        raise ValueError(
            f"{timeframe}: Duplicate timestamps found"
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if not data.empty:
        if not (
            data["high"] >= data[["open", "close"]].max(axis=1)
        ).all():
            raise ValueError(
                f"{timeframe}: Invalid high prices"
            )

        if not (
            data["low"] <= data[["open", "close"]].min(axis=1)
        ).all():
            raise ValueError(
                f"{timeframe}: Invalid low prices"
            )

    return data


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


def parse_signal_events(event_value):
    if pd.isna(event_value):
        return []

    event_text = str(event_value).strip()

    if event_text in ("", "NONE"):
        return []

    events = []

    for event_name in event_text.split("|"):
        event_name = event_name.strip()

        if event_name not in SIGNAL_TYPES:
            raise ValueError(
                f"Unsupported structure event: {event_name}"
            )

        events.append(event_name)

    return events


def calculate_excursions(
    data,
    entry_index,
    exit_index,
    direction,
    entry_price,
    risk_per_unit,
):
    trade_candles = data.iloc[entry_index:exit_index + 1]

    highest_price = float(trade_candles["high"].max())
    lowest_price = float(trade_candles["low"].min())

    if direction == 1:
        favorable_move = max(0.0, highest_price - entry_price)
        adverse_move = max(0.0, entry_price - lowest_price)
    else:
        favorable_move = max(0.0, entry_price - lowest_price)
        adverse_move = max(0.0, highest_price - entry_price)

    return {
        "mfe_R": favorable_move / risk_per_unit,
        "mae_R": adverse_move / risk_per_unit,
        "mfe_pct": (favorable_move / entry_price) * 100,
        "mae_pct": (adverse_move / entry_price) * 100,
    }


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
    initial_stop_price = float(stop_price)

    if not math.isfinite(entry_price) or not math.isfinite(initial_stop_price):
        return None, "INVALID_PRICE"

    if entry_price <= 0 or initial_stop_price <= 0:
        return None, "INVALID_PRICE"

    if direction == 1 and initial_stop_price >= entry_price:
        return None, "INVALID_LONG_STOP"

    if direction == -1 and initial_stop_price <= entry_price:
        return None, "INVALID_SHORT_STOP"

    risk_per_unit = abs(entry_price - initial_stop_price)

    if risk_per_unit <= 0:
        return None, "ZERO_RISK"

    if direction == 1:
        target_price = entry_price + RISK_REWARD * risk_per_unit
        profit_lock_price = (
            entry_price + PROFIT_LOCK_TRIGGER_R * risk_per_unit
        )
    else:
        target_price = entry_price - RISK_REWARD * risk_per_unit
        profit_lock_price = (
            entry_price - PROFIT_LOCK_TRIGGER_R * risk_per_unit
        )

    active_stop_price = initial_stop_price
    profit_lock_activated = False
    activate_lock_next_candle = False

    exit_price = None
    exit_reason = None
    exit_index = None

    end_index = entry_index + max_holding_candles

    for current_index in range(entry_index, end_index):
        candle = data.iloc[current_index]

        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])

        if activate_lock_next_candle:
            active_stop_price = profit_lock_price
            profit_lock_activated = True
            activate_lock_next_candle = False

        if direction == 1:
            if open_price <= active_stop_price:
                exit_price = open_price
                exit_reason = (
                    "PROFIT_LOCK_GAP"
                    if profit_lock_activated else "STOP_GAP"
                )

            elif open_price >= target_price:
                exit_price = open_price
                exit_reason = "TARGET_GAP"

            else:
                stop_hit = low_price <= active_stop_price
                target_hit = high_price >= target_price

                if stop_hit:
                    exit_price = active_stop_price
                    exit_reason = (
                        "PROFIT_LOCK"
                        if profit_lock_activated
                        else (
                            "STOP_BOTH_HIT"
                            if target_hit else "STOP"
                        )
                    )

                elif target_hit:
                    exit_price = target_price
                    exit_reason = "TARGET"

        else:
            if open_price >= active_stop_price:
                exit_price = open_price
                exit_reason = (
                    "PROFIT_LOCK_GAP"
                    if profit_lock_activated else "STOP_GAP"
                )

            elif open_price <= target_price:
                exit_price = open_price
                exit_reason = "TARGET_GAP"

            else:
                stop_hit = high_price >= active_stop_price
                target_hit = low_price <= target_price

                if stop_hit:
                    exit_price = active_stop_price
                    exit_reason = (
                        "PROFIT_LOCK"
                        if profit_lock_activated
                        else (
                            "STOP_BOTH_HIT"
                            if target_hit else "STOP"
                        )
                    )

                elif target_hit:
                    exit_price = target_price
                    exit_reason = "TARGET"

        if exit_price is not None:
            exit_index = current_index
            break

        if use_profit_lock and not profit_lock_activated:
            if direction == 1 and high_price >= profit_lock_price:
                activate_lock_next_candle = True

            elif direction == -1 and low_price <= profit_lock_price:
                activate_lock_next_candle = True

    if exit_price is None:
        exit_index = end_index - 1
        exit_price = float(data.iloc[exit_index]["close"])
        exit_reason = "TIME_EXIT"

    gross_pnl, net_pnl, entry_fill = calculate_trade_costs(
        direction,
        entry_price,
        exit_price,
    )

    gross_r = gross_pnl / risk_per_unit
    net_r = net_pnl / risk_per_unit
    net_return_pct = (net_pnl / entry_fill) * 100

    excursions = calculate_excursions(
        data,
        entry_index,
        exit_index,
        direction,
        entry_price,
        risk_per_unit,
    )

    trade = {
        "signal_timestamp": int(data.iloc[signal_index]["timestamp"]),
        "entry_timestamp": int(data.iloc[entry_index]["timestamp"]),
        "exit_timestamp": int(data.iloc[exit_index]["timestamp"]),
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry_price": entry_price,
        "stop_price": initial_stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_R": gross_r,
        "net_R": net_r,
        "net_return_pct": net_return_pct,
        "risk_per_unit": risk_per_unit,
        "holding_candles": exit_index - entry_index + 1,
        "profit_lock_activated": profit_lock_activated,
        "profit_lock_price": (
            profit_lock_price if profit_lock_activated else None
        ),
        "mfe_R": excursions["mfe_R"],
        "mae_R": excursions["mae_R"],
        "mfe_pct": excursions["mfe_pct"],
        "mae_pct": excursions["mae_pct"],
    }

    return trade, None


def calculate_summary(trades):
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0,
            "avg_net_R": 0,
            "median_net_R": 0,
            "profit_factor": 0,
            "avg_mfe_R": 0,
            "avg_mae_R": 0,
            "target_exits": 0,
            "stop_exits": 0,
            "profit_lock_exits": 0,
            "time_exits": 0,
        }

    frame = pd.DataFrame(trades)
    net_r = frame["net_R"]

    winners = net_r[net_r > 0]
    losers = net_r[net_r < 0]

    if len(losers):
        profit_factor = winners.sum() / abs(losers.sum())
    else:
        profit_factor = float("inf") if len(winners) else 0

    return {
        "trades": len(frame),
        "win_rate_pct": round((net_r > 0).mean() * 100, 2),
        "avg_net_R": round(net_r.mean(), 4),
        "median_net_R": round(net_r.median(), 4),
        "profit_factor": round(profit_factor, 4)
        if math.isfinite(profit_factor) else "INF",
        "avg_mfe_R": round(frame["mfe_R"].mean(), 4),
        "avg_mae_R": round(frame["mae_R"].mean(), 4),
        "target_exits": int(
            frame["exit_reason"].str.startswith("TARGET").sum()
        ),
        "stop_exits": int(
            frame["exit_reason"].str.startswith("STOP").sum()
        ),
        "profit_lock_exits": int(
            frame["exit_reason"].str.startswith("PROFIT_LOCK").sum()
        ),
        "time_exits": int(
            (frame["exit_reason"] == "TIME_EXIT").sum()
        ),
    }


def validate_structure_output(structure, timeframe):
    required_columns = set()

    for config in STRUCTURE_CONFIG.values():
        required_columns.add(config["event_column"])
        required_columns.add(config["swing_high_column"])
        required_columns.add(config["swing_low_column"])

    missing = required_columns - set(structure.columns)

    if missing:
        raise ValueError(
            f"{timeframe}: Missing structure columns: {sorted(missing)}"
        )

    return structure


def build_structure(data, timeframe, detect_market_structure):
    structure = detect_market_structure(
        data.drop(columns=["timestamp_seconds"], errors="ignore"),
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
    )

    if len(structure) != len(data):
        raise ValueError(
            f"{timeframe}: Structure output length mismatch"
        )

    return validate_structure_output(structure, timeframe)


def get_daily_bias(
    daily_data,
    daily_structure,
    decision_timestamp,
):
    daily_duration = TIMEFRAME_SECONDS["1D"]

    eligible_indices = daily_data.index[
        (
            daily_data["timestamp_seconds"] + daily_duration
        ) <= decision_timestamp
    ]

    if len(eligible_indices) == 0:
        return "WAIT"

    last_index = int(eligible_indices[-1])

    trend_column = None

    for column in DAILY_TREND_COLUMNS:
        if column in daily_structure.columns:
            trend_column = column
            break

    if trend_column is None:
        raise ValueError(
            "Daily structure has no recognized trend column. "
            "Expected one of: "
            + ", ".join(DAILY_TREND_COLUMNS)
        )

    trend_value = daily_structure.iloc[last_index][trend_column]

    if pd.isna(trend_value):
        return "WAIT"

    normalized = str(trend_value).strip().upper()

    if normalized in ("BULLISH", "BULL", "UP", "LONG", "1"):
        return "LONG"

    if normalized in ("BEARISH", "BEAR", "DOWN", "SHORT", "-1"):
        return "SHORT"

    return "WAIT"


def get_latest_confirmed_swings(structure):
    latest_swings = {}

    for structure_name, config in STRUCTURE_CONFIG.items():
        latest_swings[structure_name] = {
            "high": structure[config["swing_high_column"]].ffill(),
            "low": structure[config["swing_low_column"]].ffill(),
        }

    return latest_swings


def get_event_available_timestamp(
    structure,
    index,
    structure_name,
    candle_timestamp,
    timeframe,
):
    candidates = [
        f"{structure_name.lower()}_structure_event_available_timestamp",
        f"{structure_name.lower()}_available_timestamp",
        "available_timestamp",
    ]

    for column in candidates:
        if column not in structure.columns:
            continue

        value = structure.iloc[index][column]

        if pd.isna(value):
            continue

        value = float(value)

        if value > 100000000000:
            value /= 1000.0

        return int(value)

    return int(
        candle_timestamp + TIMEFRAME_SECONDS[timeframe]
    )


def collect_lower_timeframe_events(
    entry_data,
    entry_structure,
    timeframe,
):
    events = []

    for index in range(len(entry_data)):
        for structure_name, config in STRUCTURE_CONFIG.items():
            event_value = entry_structure.iloc[index][
                config["event_column"]
            ]

            for event_name in parse_signal_events(event_value):
                events.append({
                    "timeframe": timeframe,
                    "index": index,
                    "structure_type": structure_name,
                    "signal_type": event_name,
                    "direction": SIGNAL_TYPES[event_name],
                    "available_timestamp": (
                        get_event_available_timestamp(
                            entry_structure,
                            index,
                            structure_name,
                            int(
                                entry_data.iloc[index][
                                    "timestamp_seconds"
                                ]
                            ),
                            timeframe,
                        )
                    ),
                })

    return events


def get_holding_candles(entry_timeframe, four_hour_horizon):
    entry_seconds = TIMEFRAME_SECONDS[entry_timeframe]
    base_seconds = TIMEFRAME_SECONDS["4H"]

    holding_seconds = four_hour_horizon * base_seconds

    return int(holding_seconds // entry_seconds)


def find_entry_confirmation(
    setup,
    lower_events,
    setup_expiry_timestamp,
):
    candidates = []

    timeframe_priority = {
        "1H": 0,
        "15m": 1,
        "5m": 2,
    }

    for event in lower_events:
        if event["direction"] != setup["direction"]:
            continue

        if event["available_timestamp"] <= setup["available_timestamp"]:
            continue

        if event["available_timestamp"] > setup_expiry_timestamp:
            continue

        candidates.append(event)

    if not candidates:
        return None

    candidates.sort(
        key=lambda event: (
            event["available_timestamp"],
            timeframe_priority[event["timeframe"]],
        )
    )

    return candidates[0]


def create_4h_setups(
    four_hour_data,
    four_hour_structure,
    daily_data,
    daily_structure,
):
    setups = []
    rejected = {}

    latest_swings = get_latest_confirmed_swings(
        four_hour_structure
    )

    for index in range(len(four_hour_data)):
        candle_timestamp = int(
            four_hour_data.iloc[index]["timestamp_seconds"]
        )

        setup_available_timestamp = (
            candle_timestamp + TIMEFRAME_SECONDS["4H"]
        )

        daily_bias = get_daily_bias(
            daily_data,
            daily_structure,
            setup_available_timestamp,
        )

        if daily_bias == "WAIT":
            rejected["DAILY_BIAS_UNAVAILABLE_OR_NEUTRAL"] = (
                rejected.get(
                    "DAILY_BIAS_UNAVAILABLE_OR_NEUTRAL",
                    0,
                ) + 1
            )
            continue

        allowed_direction = (
            1 if daily_bias == "LONG" else -1
        )

        for structure_name, config in STRUCTURE_CONFIG.items():
            event_value = four_hour_structure.iloc[index][
                config["event_column"]
            ]

            for event_name in parse_signal_events(event_value):
                direction = SIGNAL_TYPES[event_name]

                if direction != allowed_direction:
                    rejected["DAILY_DIRECTION_CONFLICT"] = (
                        rejected.get(
                            "DAILY_DIRECTION_CONFLICT",
                            0,
                        ) + 1
                    )
                    continue

                if direction == 1:
                    stop_price = latest_swings[
                        structure_name
                    ]["low"].iloc[index]
                else:
                    stop_price = latest_swings[
                        structure_name
                    ]["high"].iloc[index]

                if pd.isna(stop_price):
                    rejected["NO_CONFIRMED_4H_OPPOSITE_SWING"] = (
                        rejected.get(
                            "NO_CONFIRMED_4H_OPPOSITE_SWING",
                            0,
                        ) + 1
                    )
                    continue

                event_available_timestamp = (
                    get_event_available_timestamp(
                        four_hour_structure,
                        index,
                        structure_name,
                        candle_timestamp,
                        "4H",
                    )
                )

                setup_id = (
                    f"4H_{candle_timestamp}_"
                    f"{structure_name}_{event_name}"
                )

                setups.append({
                    "setup_id": setup_id,
                    "setup_index": index,
                    "setup_timestamp": candle_timestamp,
                    "available_timestamp": (
                        event_available_timestamp
                    ),
                    "direction": direction,
                    "daily_bias": daily_bias,
                    "structure_type": structure_name,
                    "signal_type": event_name,
                    "stop_price": float(stop_price),
                })

    setups.sort(
        key=lambda setup: (
            setup["available_timestamp"],
            setup["setup_id"],
        )
    )

    return setups, rejected


def analyze_hierarchical_strategy(detect_market_structure):
    print("\n" + "=" * 70)
    print("HIERARCHICAL MULTI-TIMEFRAME BACKTEST")
    print("=" * 70)

    print("Direction filter: 1D")
    print("Setup timeframe: 4H")
    print("Entry refinement: 1H, 15m, 5m")
    print("Entry confirmation: First aligned BOS/CHOCH")
    print("Stop source: Confirmed 4H opposite swing")
    print("Base holding period: 24 equivalent 4H candles")
    print("Setup expiry: 24 4H candles")
    print("Trading execution: DISABLED")
    print("Archive modification: NO")

    archives = {}
    structures = {}

    for timeframe in TIMEFRAMES:
        archives[timeframe] = load_archive(timeframe)
        structures[timeframe] = build_structure(
            archives[timeframe],
            timeframe,
            detect_market_structure,
        )

        print(
            f"{timeframe}: loaded "
            f"{len(archives[timeframe])} candles"
        )

    setups, rejected_setups = create_4h_setups(
        archives["4H"],
        structures["4H"],
        archives["1D"],
        structures["1D"],
    )

    print(f"\nDaily-filtered 4H setups: {len(setups)}")
    print(f"Rejected setup events: {sum(rejected_setups.values())}")

    for reason, count in sorted(rejected_setups.items()):
        print(f"  {reason}: {count}")

    lower_events = []

    for timeframe in ENTRY_TIMEFRAMES:
        events = collect_lower_timeframe_events(
            archives[timeframe],
            structures[timeframe],
            timeframe,
        )

        lower_events.extend(events)

        print(
            f"{timeframe} aligned-entry event pool: "
            f"{len(events)}"
        )

    baseline_trades = []
    managed_trades = []
    extensions = []

    statistics = {
        "valid_4h_setups": len(setups),
        "entry_confirmed": 0,
        "pending_or_expired": 0,
        "insufficient_future_data": 0,
        "invalid_trades": 0,
        "used_setup_ids": set(),
        "entry_timeframes": {},
    }

    for setup in setups:
        expiry_timestamp = (
            setup["available_timestamp"]
            + SETUP_MAX_WAIT_4H_CANDLES
            * TIMEFRAME_SECONDS["4H"]
        )

        entry_event = find_entry_confirmation(
            setup,
            lower_events,
            expiry_timestamp,
        )

        if entry_event is None:
            statistics["pending_or_expired"] += 1
            continue

        if setup["setup_id"] in statistics["used_setup_ids"]:
            continue

        timeframe = entry_event["timeframe"]
        entry_data = archives[timeframe]
        entry_index = entry_event["index"]

        holding_candles = get_holding_candles(
            timeframe,
            BASE_HOLDING_CANDLES,
        )

        base_trade, reason = simulate_trade(
            entry_data,
            entry_index,
            setup["direction"],
            setup["stop_price"],
            holding_candles,
            use_profit_lock=False,
        )

        if reason:
            if reason == "INSUFFICIENT_FUTURE_DATA":
                statistics["insufficient_future_data"] += 1
            else:
                statistics["invalid_trades"] += 1
            continue

        managed_trade, managed_reason = simulate_trade(
            entry_data,
            entry_index,
            setup["direction"],
            setup["stop_price"],
            holding_candles,
            use_profit_lock=True,
        )

        if managed_reason:
            statistics["invalid_trades"] += 1
            continue

        metadata = {
            "setup_id": setup["setup_id"],
            "setup_timeframe": "4H",
            "setup_timestamp": setup["setup_timestamp"],
            "setup_type": setup["structure_type"],
            "setup_signal_type": setup["signal_type"],
            "daily_bias": setup["daily_bias"],
            "entry_timeframe": timeframe,
            "entry_structure_type": entry_event["structure_type"],
            "entry_signal_type": entry_event["signal_type"],
            "setup_expiry_timestamp": expiry_timestamp,
            "holding_period_4h_equivalent": BASE_HOLDING_CANDLES,
        }

        for trade in [base_trade, managed_trade]:
            trade.update(metadata)
            trade["timeframe"] = timeframe
            trade["structure_type"] = setup["structure_type"]
            trade["signal_type"] = setup["signal_type"]

        baseline_trades.append(base_trade)
        managed_trades.append(managed_trade)

        statistics["used_setup_ids"].add(setup["setup_id"])
        statistics["entry_confirmed"] += 1

        statistics["entry_timeframes"][timeframe] = (
            statistics["entry_timeframes"].get(timeframe, 0) + 1
        )

        if base_trade["exit_reason"] == "TIME_EXIT":
            for horizon in EXTENDED_HORIZONS:
                extended_holding = get_holding_candles(
                    timeframe,
                    horizon,
                )

                extended_trade, extension_reason = simulate_trade(
                    entry_data,
                    entry_index,
                    setup["direction"],
                    setup["stop_price"],
                    extended_holding,
                    use_profit_lock=False,
                )

                if extension_reason:
                    extensions.append({
                        **metadata,
                        "timeframe": timeframe,
                        "horizon_4h_equivalent": horizon,
                        "status": extension_reason,
                    })
                    continue

                extended_trade.update(metadata)
                extended_trade["timeframe"] = timeframe
                extended_trade["structure_type"] = setup["structure_type"]
                extended_trade["signal_type"] = setup["signal_type"]
                extended_trade["horizon"] = horizon
                extended_trade["status"] = "AVAILABLE"

                extensions.append(extended_trade)

    statistics.pop("used_setup_ids")

    print("\n" + "=" * 70)
    print("HIERARCHICAL BACKTEST RESULTS")
    print("=" * 70)

    print(f"Valid 4H setups: {statistics['valid_4h_setups']}")
    print(f"Entry-confirmed setups: {statistics['entry_confirmed']}")
    print(f"Pending or expired setups: {statistics['pending_or_expired']}")
    print(
        "Insufficient future data: "
        f"{statistics['insufficient_future_data']}"
    )
    print(f"Invalid trades: {statistics['invalid_trades']}")

    print("\nEntries by timeframe:")
    print(statistics["entry_timeframes"])

    print("\nBaseline summary:")
    print(calculate_summary(baseline_trades))

    print("\nProfit Lock summary:")
    print(calculate_summary(managed_trades))

    if baseline_trades:
        baseline_frame = pd.DataFrame(baseline_trades)
        managed_frame = pd.DataFrame(managed_trades)

        print_grouped_summary(
            baseline_frame,
            ["entry_timeframe"],
            "BASELINE BY ENTRY TIMEFRAME",
        )

        print_grouped_summary(
            managed_frame,
            ["entry_timeframe"],
            "PROFIT LOCK BY ENTRY TIMEFRAME",
        )

        print_grouped_summary(
            managed_frame,
            ["daily_bias", "setup_signal_type"],
            "PROFIT LOCK BY DAILY BIAS AND 4H SETUP",
        )

        print("\nProfit Lock exit reasons:")
        print(
            managed_frame["exit_reason"].value_counts().to_dict()
        )

    print_extension_report(extensions)

    print("\nIMPORTANT LIMITATIONS")
    print("- Daily direction is evaluated using closed daily candles only.")
    print("- Only 4H structure events can create trade setups.")
    print("- Lower timeframes are used only for entry confirmation.")
    print("- One setup ID can produce no more than one trade.")
    print("- Stop prices come from confirmed 4H opposite swings.")
    print("- Holding periods are measured in equivalent 4H candles.")
    print("- OHLC data cannot determine exact intrabar price ordering.")
    print("- Same-candle stop and target conflicts remain conservative.")
    print("- Fees and slippage are applied to every simulated trade.")
    print("- No position sizing or capital compounding is applied.")

    print("\nHIERARCHICAL BACKTEST COMPLETED")

    return {
        "setups": setups,
        "baseline_trades": baseline_trades,
        "managed_trades": managed_trades,
        "extensions": extensions,
        "statistics": statistics,
    }


def analyze_timeframe(timeframe, detect_market_structure):
    filename = f"BTC_USDT_SPOT_{timeframe}.csv"
    filepath = ARCHIVE_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Archive not found: {filepath}"
        )

    data = pd.read_csv(filepath)

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{timeframe}: Missing columns: {sorted(missing)}"
        )

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError(
            f"{timeframe}: Duplicate timestamps found"
        )

    for column in [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    structure = detect_market_structure(
        data,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
    )

    validate_structure_output(structure, timeframe)

    latest_swings = get_latest_confirmed_swings(structure)

    baseline_trades = []
    managed_trades = []
    extensions = []
    skipped = {}

    for index in range(len(data)):
        for structure_name, config in STRUCTURE_CONFIG.items():
            event_value = structure.iloc[index][config["event_column"]]
            events = parse_signal_events(event_value)

            for event_name in events:
                direction = SIGNAL_TYPES[event_name]

                if direction == 1:
                    stop_price = latest_swings[
                        structure_name
                    ]["low"].iloc[index]
                else:
                    stop_price = latest_swings[
                        structure_name
                    ]["high"].iloc[index]

                if pd.isna(stop_price):
                    reason = "NO_CONFIRMED_OPPOSITE_SWING"
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                base_trade, reason = simulate_trade(
                    data,
                    index,
                    direction,
                    stop_price,
                    BASE_HOLDING_CANDLES,
                    use_profit_lock=False,
                )

                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                managed_trade, managed_reason = simulate_trade(
                    data,
                    index,
                    direction,
                    stop_price,
                    BASE_HOLDING_CANDLES,
                    use_profit_lock=True,
                )

                if managed_reason:
                    skipped[managed_reason] = (
                        skipped.get(managed_reason, 0) + 1
                    )
                    continue

                for trade in [base_trade, managed_trade]:
                    trade["timeframe"] = timeframe
                    trade["structure_type"] = structure_name
                    trade["signal_type"] = event_name

                baseline_trades.append(base_trade)
                managed_trades.append(managed_trade)

                if base_trade["exit_reason"] != "TIME_EXIT":
                    continue

                for horizon in EXTENDED_HORIZONS:
                    extended_trade, extension_reason = simulate_trade(
                        data,
                        index,
                        direction,
                        stop_price,
                        horizon,
                        use_profit_lock=False,
                    )

                    if extension_reason:
                        extensions.append({
                            "timeframe": timeframe,
                            "structure_type": structure_name,
                            "signal_type": event_name,
                            "horizon": horizon,
                            "status": extension_reason,
                        })
                        continue

                    extended_trade["timeframe"] = timeframe
                    extended_trade["structure_type"] = structure_name
                    extended_trade["signal_type"] = event_name
                    extended_trade["horizon"] = horizon
                    extended_trade["status"] = "AVAILABLE"

                    extensions.append(extended_trade)

    return baseline_trades, managed_trades, extensions, skipped


def print_grouped_summary(frame, group_columns, heading):
    print(f"\n{heading}")

    if frame.empty:
        print("No trades available.")
        return

    for group_key, group in frame.groupby(group_columns):
        summary = calculate_summary(group.to_dict("records"))

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        label = " | ".join(str(value) for value in group_key)
        print(f"{label}: {summary}")


def print_extension_report(extensions):
    print("\nBASELINE EXTENDED HOLDING ANALYSIS")

    if not extensions:
        print("No Time Exit trades available for extension analysis.")
        return

    extension_frame = pd.DataFrame(extensions)

    for horizon in EXTENDED_HORIZONS:
        if "horizon" not in extension_frame.columns:
            continue

        subset = extension_frame[
            (extension_frame["horizon"] == horizon)
            & (extension_frame["status"] == "AVAILABLE")
        ]

        unavailable = extension_frame[
            (extension_frame["horizon"] == horizon)
            & (extension_frame["status"] != "AVAILABLE")
        ]

        print(f"\nHORIZON: {horizon} CANDLES")
        print(f"Available extended trades: {len(subset)}")
        print(f"Unavailable due to data limits: {len(unavailable)}")

        if subset.empty:
            continue

        print("Extended outcome summary:")
        print(calculate_summary(subset.to_dict("records")))

        print("Extended exit reasons:")
        print(subset["exit_reason"].value_counts().to_dict())

        print("Results by timeframe:")
        for timeframe, group in subset.groupby("timeframe"):
            print(
                f"{timeframe}: "
                f"{calculate_summary(group.to_dict('records'))}"
            )

        print("Results by structure:")
        for structure_type, group in subset.groupby("structure_type"):
            print(
                f"{structure_type}: "
                f"{calculate_summary(group.to_dict('records'))}"
            )

    print("\nEXTENSION COMPARISON")
    print(
        "Only baseline trades that reached TIME_EXIT at the base "
        "holding horizon are included."
    )
    print(
        "Extended simulations restart from the original entry "
        "with the original stop and target."
    )


def main():
    print("=" * 70)
    print("BTC HIERARCHICAL MARKET STRUCTURE BACKTEST")
    print("=" * 70)

    print(f"Setup timeframe: {SETUP_TIMEFRAME}")
    print(f"Direction timeframe: {DIRECTION_TIMEFRAME}")
    print(f"Entry timeframes: {', '.join(ENTRY_TIMEFRAMES)}")
    print(f"Risk/Reward: 1:{RISK_REWARD}")
    print(f"Base holding period: {BASE_HOLDING_CANDLES} 4H candles")
    print(f"Setup expiry: {SETUP_MAX_WAIT_4H_CANDLES} 4H candles")
    print(f"Profit lock trigger: +{PROFIT_LOCK_TRIGGER_R}R")
    print(f"Extended horizons: {EXTENDED_HORIZONS}")
    print(f"Fee per side: {FEE_RATE * 100:.3f}%")
    print(f"Slippage per side: {SLIPPAGE_RATE * 100:.3f}%")
    print("Entry: Next lower-timeframe candle Open")
    print("Same-candle SL/TP: Stop Loss First")
    print("Profit lock activation: Next candle after +0.5R trigger")
    print("Trading execution: DISABLED")
    print("Archive modification: NO")

    detect_market_structure = load_structure_module()

    analyze_hierarchical_strategy(
        detect_market_structure
    )


if __name__ == "__main__":
    main()
