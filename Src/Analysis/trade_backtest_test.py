
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
        "available_column": "internal_event_available_timestamp",
    },
    "EXTERNAL": {
        "event_column": "external_structure_event",
        "swing_high_column": "external_swing_high_price",
        "swing_low_column": "external_swing_low_price",
        "available_column": "external_event_available_timestamp",
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

    data["timestamp"] = pd.to_numeric(
        data["timestamp"],
        errors="raise",
    )

    median_timestamp = float(data["timestamp"].median())

    if median_timestamp > 100000000000:
        data["timestamp_seconds"] = (
            data["timestamp"] / 1000.0
        ).astype("int64")
    else:
        data["timestamp_seconds"] = (
            data["timestamp"].astype("int64")
        )

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

        if (data["low"] <= 0).any():
            raise ValueError(
                f"{timeframe}: Non-positive low prices found"
            )

    return data


def build_structure(data, timeframe, detect_market_structure):
    input_data = data.drop(
        columns=["timestamp_seconds"],
        errors="ignore",
    ).copy()

    structure = detect_market_structure(
        input_data,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
        timeframe=timeframe,
    )

    if len(structure) != len(data):
        raise ValueError(
            f"{timeframe}: Structure output length mismatch"
        )

    required_columns = set()

    for config in STRUCTURE_CONFIG.values():
        required_columns.add(config["event_column"])
        required_columns.add(config["swing_high_column"])
        required_columns.add(config["swing_low_column"])
        required_columns.add(config["available_column"])

    missing = required_columns - set(structure.columns)

    if missing:
        raise ValueError(
            f"{timeframe}: Missing structure columns: "
            f"{sorted(missing)}"
        )

    if DIRECTION_COLUMN not in structure.columns:
        raise ValueError(
            f"{timeframe}: Missing direction column "
            f"{DIRECTION_COLUMN}"
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

    events = []

    for event_name in event_text.split("|"):
        event_name = event_name.strip()

        if event_name in SIGNAL_TYPES:
            events.append(event_name)

    return events


def get_event_available_timestamp(
    structure,
    index,
    structure_name,
):
    column = STRUCTURE_CONFIG[structure_name]["available_column"]

    value = structure.iloc[index][column]

    if pd.isna(value):
        return None

    return normalize_timestamp_value(value)


def collect_structure_events(data, structure, timeframe):
    events = []

    for index in range(len(data)):
        for structure_name, config in STRUCTURE_CONFIG.items():
            event_value = structure.iloc[index][
                config["event_column"]
            ]

            event_names = parse_signal_events(event_value)

            if not event_names:
                continue

            available_timestamp = get_event_available_timestamp(
                structure,
                index,
                structure_name,
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
                })

    events.sort(
        key=lambda item: (
            item["available_timestamp"],
            item["index"],
            item["structure_type"],
        )
    )

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

    if normalized in (
        "1",
        "BULLISH",
        "BULL",
        "UP",
        "LONG",
    ):
        return "LONG"

    if normalized in (
        "-1",
        "BEARISH",
        "BEAR",
        "DOWN",
        "SHORT",
    ):
        return "SHORT"

    if normalized in (
        "0",
        "SIDEWAYS",
        "NEUTRAL",
        "NONE",
        "WAIT",
    ):
        return "WAIT"

    return "WAIT"


def get_daily_bias(
    daily_data,
    daily_structure,
    decision_timestamp,
):
    daily_duration = TIMEFRAME_SECONDS["1D"]

    eligible_indices = daily_data.index[
        (
            daily_data["timestamp_seconds"]
            + daily_duration
        ) <= decision_timestamp
    ]

    if len(eligible_indices) == 0:
        return "WAIT"

    last_index = int(eligible_indices[-1])

    trend_value = daily_structure.iloc[last_index][
        DIRECTION_COLUMN
    ]

    return normalize_daily_direction(trend_value)


# ============================================================
# TRADE COSTS AND EXCURSIONS
# ============================================================

def calculate_trade_costs(
    direction,
    entry_price,
    exit_price,
):
    if direction == 1:
        entry_fill = entry_price * (1 + SLIPPAGE_RATE)
        exit_fill = exit_price * (1 - SLIPPAGE_RATE)

        gross_pnl = exit_price - entry_price

        net_pnl = (
            exit_fill
            - entry_fill
            - FEE_RATE * (entry_fill + exit_fill)
        )

    else:
        entry_fill = entry_price * (1 - SLIPPAGE_RATE)
        exit_fill = exit_price * (1 + SLIPPAGE_RATE)

        gross_pnl = entry_price - exit_price

        net_pnl = (
            entry_fill
            - exit_fill
            - FEE_RATE * (entry_fill + exit_fill)
        )

    return gross_pnl, net_pnl, entry_fill


def calculate_excursions(
    data,
    entry_index,
    exit_index,
    direction,
    entry_price,
    risk_per_unit,
):
    trade_candles = data.iloc[
        entry_index:exit_index + 1
    ]

    highest_price = float(trade_candles["high"].max())
    lowest_price = float(trade_candles["low"].min())

    if direction == 1:
        favorable_move = max(
            0.0,
            highest_price - entry_price,
        )

        adverse_move = max(
            0.0,
            entry_price - lowest_price,
        )

    else:
        favorable_move = max(
            0.0,
            entry_price - lowest_price,
        )

        adverse_move = max(
            0.0,
            highest_price - entry_price,
        )

    return {
        "mfe_R": favorable_move / risk_per_unit,
        "mae_R": adverse_move / risk_per_unit,
        "mfe_pct": (
            favorable_move / entry_price
        ) * 100,
        "mae_pct": (
            adverse_move / entry_price
        ) * 100,
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

    entry_price = float(
        data.iloc[entry_index]["open"]
    )

    initial_stop_price = float(stop_price)

    if (
        not math.isfinite(entry_price)
        or not math.isfinite(initial_stop_price)
    ):
        return None, "INVALID_PRICE"

    if entry_price <= 0 or initial_stop_price <= 0:
        return None, "INVALID_PRICE"

    if direction == 1 and initial_stop_price >= entry_price:
        return None, "INVALID_LONG_STOP"

    if direction == -1 and initial_stop_price <= entry_price:
        return None, "INVALID_SHORT_STOP"

    risk_per_unit = abs(
        entry_price - initial_stop_price
    )

    if risk_per_unit <= 0:
        return None, "ZERO_RISK"

    if direction == 1:
        target_price = (
            entry_price + RISK_REWARD * risk_per_unit
        )

        profit_lock_price = (
            entry_price
            + PROFIT_LOCK_TRIGGER_R * risk_per_unit
        )

    else:
        target_price = (
            entry_price - RISK_REWARD * risk_per_unit
        )

        profit_lock_price = (
            entry_price
            - PROFIT_LOCK_TRIGGER_R * risk_per_unit
        )

    active_stop_price = initial_stop_price

    profit_lock_activated = False
    activate_lock_next_candle = False

    exit_price = None
    exit_reason = None
    exit_index = None

    end_index = entry_index + max_holding_candles

    for current_index in range(
        entry_index,
        end_index,
    ):
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
                    if profit_lock_activated
                    else "STOP_GAP"
                )

            elif open_price >= target_price:
                exit_price = open_price
                exit_reason = "TARGET_GAP"

            else:
                stop_hit = low_price <= active_stop_price
                target_hit = high_price >= target_price

                if stop_hit:
                    exit_price = active_stop_price

                    if profit_lock_activated:
                        exit_reason = "PROFIT_LOCK"

                    elif target_hit:
                        exit_reason = "STOP_BOTH_HIT"

                    else:
                        exit_reason = "STOP"

                elif target_hit:
                    exit_price = target_price
                    exit_reason = "TARGET"

        else:
            if open_price >= active_stop_price:
                exit_price = open_price

                exit_reason = (
                    "PROFIT_LOCK_GAP"
                    if profit_lock_activated
                    else "STOP_GAP"
                )

            elif open_price <= target_price:
                exit_price = open_price
                exit_reason = "TARGET_GAP"

            else:
                stop_hit = high_price >= active_stop_price
                target_hit = low_price <= target_price

                if stop_hit:
                    exit_price = active_stop_price

                    if profit_lock_activated:
                        exit_reason = "PROFIT_LOCK"

                    elif target_hit:
                        exit_reason = "STOP_BOTH_HIT"

                    else:
                        exit_reason = "STOP"

                elif target_hit:
                    exit_price = target_price
                    exit_reason = "TARGET"

        if exit_price is not None:
            exit_index = current_index
            break

        if use_profit_lock and not profit_lock_activated:
            if (
                direction == 1
                and high_price >= profit_lock_price
            ):
                activate_lock_next_candle = True

            elif (
                direction == -1
                and low_price <= profit_lock_price
            ):
                activate_lock_next_candle = True

    if exit_price is None:
        exit_index = end_index - 1

        exit_price = float(
            data.iloc[exit_index]["close"]
        )

        exit_reason = "TIME_EXIT"

    gross_pnl, net_pnl, entry_fill = calculate_trade_costs(
        direction,
        entry_price,
        exit_price,
    )

    gross_r = gross_pnl / risk_per_unit
    net_r = net_pnl / risk_per_unit

    net_return_pct = (
        net_pnl / entry_fill
    ) * 100

    excursions = calculate_excursions(
        data,
        entry_index,
        exit_index,
        direction,
        entry_price,
        risk_per_unit,
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
        "direction": (
            "LONG" if direction == 1 else "SHORT"
        ),
        "entry_price": entry_price,
        "stop_price": initial_stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_R": gross_r,
        "net_R": net_r,
        "net_return_pct": net_return_pct,
        "risk_per_unit": risk_per_unit,
        "holding_candles": (
            exit_index - entry_index + 1
        ),
        "profit_lock_activated": profit_lock_activated,
        "profit_lock_price": (
            profit_lock_price
            if profit_lock_activated
            else None
        ),
        "mfe_R": excursions["mfe_R"],
        "mae_R": excursions["mae_R"],
        "mfe_pct": excursions["mfe_pct"],
        "mae_pct": excursions["mae_pct"],
    }

    return trade, None


# ============================================================
# 4H SETUP GENERATION
# ============================================================

def get_latest_confirmed_swings(structure):
    latest_swings = {}

    for structure_name, config in STRUCTURE_CONFIG.items():
        latest_swings[structure_name] = {
            "high": structure[
                config["swing_high_column"]
            ].ffill(),

            "low": structure[
                config["swing_low_column"]
            ].ffill(),
        }

    return latest_swings


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

    four_hour_events = collect_structure_events(
        four_hour_data,
        four_hour_structure,
        "4H",
    )

    for event in four_hour_events:
        index = event["index"]

        setup_available_timestamp = (
            event["available_timestamp"]
        )

        daily_bias = get_daily_bias(
            daily_data,
            daily_structure,
            setup_available_timestamp,
        )

        if daily_bias == "WAIT":
            rejected["DAILY_BIAS_WAIT"] = (
                rejected.get("DAILY_BIAS_WAIT", 0) + 1
            )
            continue

        expected_direction = (
            1 if daily_bias == "LONG" else -1
        )

        if event["direction"] != expected_direction:
            rejected["DAILY_DIRECTION_CONFLICT"] = (
                rejected.get(
                    "DAILY_DIRECTION_CONFLICT",
                    0,
                ) + 1
            )
            continue

        structure_name = event["structure_type"]

        if event["direction"] == 1:
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

        setup_id = (
            f"4H_{event['available_timestamp']}_"
            f"{structure_name}_{event['signal_type']}"
        )

        setups.append({
            "setup_id": setup_id,
            "setup_index": index,
            "setup_timestamp": int(
                four_hour_data.iloc[index][
                    "timestamp_seconds"
                ]
            ),
            "available_timestamp": (
                setup_available_timestamp
            ),
            "direction": event["direction"],
            "daily_bias": daily_bias,
            "structure_type": structure_name,
            "signal_type": event["signal_type"],
            "stop_price": float(stop_price),
        })

    setups.sort(
        key=lambda setup: (
            setup["available_timestamp"],
            setup["setup_id"],
        )
    )

    return setups, rejected, four_hour_events


# ============================================================
# LOWER TIMEFRAME ENTRY CONFIRMATION
# ============================================================

def collect_lower_timeframe_events(
    archives,
    structures,
):
    all_events = []

    for timeframe in ENTRY_TIMEFRAMES:
        events = collect_structure_events(
            archives[timeframe],
            structures[timeframe],
            timeframe,
        )

        all_events.extend(events)

    timeframe_priority = {
        "1H": 0,
        "15m": 1,
        "5m": 2,
    }

    all_events.sort(
        key=lambda event: (
            event["available_timestamp"],
            timeframe_priority[event["timeframe"]],
            event["index"],
        )
    )

    return all_events


def find_entry_confirmation(
    setup,
    lower_events,
    expiry_timestamp,
    consumed_event_ids,
):
    candidates = []

    for event in lower_events:
        if event["event_id"] in consumed_event_ids:
            continue

        if event["direction"] != setup["direction"]:
            continue

        if (
            event["available_timestamp"]
            < setup["available_timestamp"]
        ):
            continue

        if event["available_timestamp"] > expiry_timestamp:
            continue

        candidates.append(event)

    if not candidates:
        return None

    timeframe_priority = {
        "1H": 0,
        "15m": 1,
        "5m": 2,
    }

    candidates.sort(
        key=lambda event: (
            event["available_timestamp"],
            timeframe_priority[event["timeframe"]],
            event["index"],
        )
    )

    return candidates[0]


# ============================================================
# SETUP INVALIDATION
# ============================================================

def is_setup_invalidated(
    setup,
    entry_event,
    four_hour_data,
    four_hour_events,
):
    setup_timestamp = setup["available_timestamp"]

    entry_timestamp = entry_event["available_timestamp"]

    direction = setup["direction"]
    stop_price = setup["stop_price"]

    # Check only fully closed 4H candles after setup confirmation.
    for index in range(len(four_hour_data)):
        candle_timestamp = int(
            four_hour_data.iloc[index][
                "timestamp_seconds"
            ]
        )

        candle_close_timestamp = (
            candle_timestamp + TIMEFRAME_SECONDS["4H"]
        )

        if candle_close_timestamp <= setup_timestamp:
            continue

        if candle_close_timestamp > entry_timestamp:
            continue

        candle = four_hour_data.iloc[index]

        if direction == 1:
            if float(candle["low"]) <= stop_price:
                return True, "STOP_TOUCHED_BEFORE_ENTRY"

        else:
            if float(candle["high"]) >= stop_price:
                return True, "STOP_TOUCHED_BEFORE_ENTRY"

    # An opposite 4H structure event invalidates the setup.
    for event in four_hour_events:
        if (
            event["available_timestamp"] <= setup_timestamp
            or event["available_timestamp"] > entry_timestamp
        ):
            continue

        if event["direction"] != direction:
            return True, "OPPOSITE_4H_STRUCTURE_EVENT"

    return False, None


# ============================================================
# HOLDING PERIOD CONVERSION
# ============================================================

def get_holding_candles(
    entry_timeframe,
    four_hour_horizon,
):
    holding_seconds = (
        four_hour_horizon
        * TIMEFRAME_SECONDS["4H"]
    )

    entry_seconds = TIMEFRAME_SECONDS[
        entry_timeframe
    ]

    return int(
        holding_seconds // entry_seconds
    )


# ============================================================
# SUMMARY REPORTS
# ============================================================

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
        profit_factor = (
            winners.sum() / abs(losers.sum())
        )
    else:
        profit_factor = (
            float("inf") if len(winners) else 0
        )

    return {
        "trades": len(frame),
        "win_rate_pct": round(
            (net_r > 0).mean() * 100,
            2,
        ),
        "avg_net_R": round(
            net_r.mean(),
            4,
        ),
        "median_net_R": round(
            net_r.median(),
            4,
        ),
        "profit_factor": (
            round(profit_factor, 4)
            if math.isfinite(profit_factor)
            else "INF"
        ),
        "avg_mfe_R": round(
            frame["mfe_R"].mean(),
            4,
        ),
        "avg_mae_R": round(
            frame["mae_R"].mean(),
            4,
        ),
        "target_exits": int(
            frame["exit_reason"].str.startswith(
                "TARGET"
            ).sum()
        ),
        "stop_exits": int(
            frame["exit_reason"].str.startswith(
                "STOP"
            ).sum()
        ),
        "profit_lock_exits": int(
            frame["exit_reason"].str.startswith(
                "PROFIT_LOCK"
            ).sum()
        ),
        "time_exits": int(
            (
                frame["exit_reason"] == "TIME_EXIT"
            ).sum()
        ),
    }


def print_grouped_summary(
    frame,
    group_columns,
    heading,
):
    print(f"\n{heading}")

    if frame.empty:
        print("No trades available.")
        return

    for group_key, group in frame.groupby(
        group_columns
    ):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        label = " | ".join(
            str(value) for value in group_key
        )

        summary = calculate_summary(
            group.to_dict("records")
        )

        print(f"{label}: {summary}")


def print_extension_report(extensions):
    print("\nBASELINE EXTENDED HOLDING ANALYSIS")

    if not extensions:
        print(
            "No eligible Time Exit trades for extension analysis."
        )
        return

    extension_frame = pd.DataFrame(extensions)

    for horizon in EXTENDED_HORIZONS:
        subset = extension_frame[
            (
                extension_frame["horizon_4h"] == horizon
            )
            & (
                extension_frame["status"] == "AVAILABLE"
            )
        ]

        unavailable = extension_frame[
            (
                extension_frame["horizon_4h"] == horizon
            )
            & (
                extension_frame["status"] != "AVAILABLE"
            )
        ]

        print(
            f"\nHORIZON: {horizon} equivalent 4H candles"
        )

        print(
            f"Available trades: {len(subset)}"
        )

        print(
            f"Unavailable trades: {len(unavailable)}"
        )

        if subset.empty:
            continue

        print(
            "Summary:",
            calculate_summary(
                subset.to_dict("records")
            ),
        )

        print(
            "Exit reasons:",
            subset["exit_reason"].value_counts().to_dict(),
        )

        print("By entry timeframe:")

        for timeframe, group in subset.groupby(
            "entry_timeframe"
        ):
            print(
                timeframe,
                calculate_summary(
                    group.to_dict("records")
                ),
            )


# ============================================================
# HIERARCHICAL BACKTEST ENGINE
# ============================================================

def analyze_hierarchical_strategy(
    detect_market_structure,
):
    print("\n" + "=" * 72)
    print("HIERARCHICAL MULTI-TIMEFRAME BACKTEST")
    print("=" * 72)

    print("Direction filter: 1D")
    print("Setup timeframe: 4H")
    print("Entry timeframes: 1H, 15m, 5m")
    print("Entry confirmation: First aligned BOS/CHOCH")
    print("Stop source: Confirmed 4H opposite swing")
    print("Base holding period: 24 equivalent 4H candles")
    print("Setup expiry: 24 equivalent 4H candles")
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
            f"{timeframe}: "
            f"{len(archives[timeframe])} candles loaded"
        )

    # --------------------------------------------------------
    # Generate daily-filtered 4H setups.
    # --------------------------------------------------------

    setups, rejected_setups, four_hour_events = (
        create_4h_setups(
            archives["4H"],
            structures["4H"],
            archives["1D"],
            structures["1D"],
        )
    )

    print(
        f"\nDaily-filtered 4H setups: {len(setups)}"
    )

    print(
        f"Rejected 4H events: {sum(rejected_setups.values())}"
    )

    for reason, count in sorted(
        rejected_setups.items()
    ):
        print(f"  {reason}: {count}")

    # --------------------------------------------------------
    # Collect lower-timeframe entry confirmations.
    # --------------------------------------------------------

    lower_events = collect_lower_timeframe_events(
        archives,
        structures,
    )

    print(
        f"\nLower-timeframe structure events: "
        f"{len(lower_events)}"
    )

    for timeframe in ENTRY_TIMEFRAMES:
        count = sum(
            event["timeframe"] == timeframe
            for event in lower_events
        )

        print(
            f"  {timeframe}: {count} events"
        )

    # --------------------------------------------------------
    # Process each 4H setup.
    # --------------------------------------------------------

    baseline_trades = []
    managed_trades = []
    extensions = []

    consumed_event_ids = set()

    statistics = {
        "valid_4h_setups": len(setups),
        "entry_confirmed": 0,
        "pending_or_expired": 0,
        "invalidated_before_entry": 0,
        "insufficient_future_data": 0,
        "invalid_trades": 0,
        "entry_timeframes": {},
        "invalidation_reasons": {},
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
            consumed_event_ids,
        )

        if entry_event is None:
            statistics["pending_or_expired"] += 1
            continue

        invalidated, invalidation_reason = (
            is_setup_invalidated(
                setup,
                entry_event,
                archives["4H"],
                four_hour_events,
            )
        )

        if invalidated:
            statistics["invalidated_before_entry"] += 1

            statistics["invalidation_reasons"][
                invalidation_reason
            ] = (
                statistics["invalidation_reasons"].get(
                    invalidation_reason,
                    0,
                ) + 1
            )

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
            "setup_available_timestamp": (
                setup["available_timestamp"]
            ),
            "setup_type": setup["structure_type"],
            "setup_signal_type": setup["signal_type"],
            "daily_bias": setup["daily_bias"],
            "entry_timeframe": timeframe,
            "entry_structure_type": (
                entry_event["structure_type"]
            ),
            "entry_signal_type": (
                entry_event["signal_type"]
            ),
            "entry_confirmation_timestamp": (
                entry_event["available_timestamp"]
            ),
            "setup_expiry_timestamp": expiry_timestamp,
            "holding_period_4h_equivalent": (
                BASE_HOLDING_CANDLES
            ),
        }

        for trade in (
            base_trade,
            managed_trade,
        ):
            trade.update(metadata)

        baseline_trades.append(base_trade)
        managed_trades.append(managed_trade)

        consumed_event_ids.add(
            entry_event["event_id"]
        )

        statistics["entry_confirmed"] += 1

        statistics["entry_timeframes"][timeframe] = (
            statistics["entry_timeframes"].get(
                timeframe,
                0,
            ) + 1
        )

        # ----------------------------------------------------
        # Extended holding analysis for baseline Time Exits.
        # ----------------------------------------------------

        if base_trade["exit_reason"] != "TIME_EXIT":
            continue

        for horizon in EXTENDED_HORIZONS:
            extended_holding = get_holding_candles(
                timeframe,
                horizon,
            )

            extended_trade, extension_reason = (
                simulate_trade(
                    entry_data,
                    entry_index,
                    setup["direction"],
                    setup["stop_price"],
                    extended_holding,
                    use_profit_lock=False,
                )
            )

            if extension_reason:
                extensions.append({
                    **metadata,
                    "entry_timeframe": timeframe,
                    "horizon_4h": horizon,
                    "status": extension_reason,
                })

                continue

            extended_trade.update(metadata)

            extended_trade["horizon_4h"] = horizon
            extended_trade["status"] = "AVAILABLE"

            extensions.append(extended_trade)

    # --------------------------------------------------------
    # Final reports.
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("HIERARCHICAL BACKTEST RESULTS")
    print("=" * 72)

    print(
        f"Valid 4H setups: "
        f"{statistics['valid_4h_setups']}"
    )

    print(
        f"Entry-confirmed setups: "
        f"{statistics['entry_confirmed']}"
    )

    print(
        f"Pending or expired setups: "
        f"{statistics['pending_or_expired']}"
    )

    print(
        f"Invalidated before entry: "
        f"{statistics['invalidated_before_entry']}"
    )

    print(
        f"Insufficient future data: "
        f"{statistics['insufficient_future_data']}"
    )

    print(
        f"Invalid trades: "
        f"{statistics['invalid_trades']}"
    )

    print("\nEntries by timeframe:")

    for timeframe in ENTRY_TIMEFRAMES:
        print(
            f"  {timeframe}: "
            f"{statistics['entry_timeframes'].get(timeframe, 0)}"
        )

    print("\nInvalidation reasons:")

    for reason, count in sorted(
        statistics["invalidation_reasons"].items()
    ):
        print(f"  {reason}: {count}")

    print("\nBASELINE SUMMARY")

    print(
        calculate_summary(baseline_trades)
    )

    print("\nPROFIT LOCK SUMMARY")

    print(
        calculate_summary(managed_trades)
    )

    if baseline_trades:
        baseline_frame = pd.DataFrame(
            baseline_trades
        )

        managed_frame = pd.DataFrame(
            managed_trades
        )

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
            [
                "daily_bias",
                "setup_signal_type",
            ],
            "PROFIT LOCK BY DAILY BIAS AND 4H SETUP",
        )

        print("\nPROFIT LOCK EXIT REASONS")

        print(
            managed_frame["exit_reason"]
            .value_counts()
            .to_dict()
        )

    print_extension_report(extensions)

    print("\n" + "=" * 72)
    print("DATA AND METHODOLOGY NOTES")
    print("=" * 72)

    five_minute_required = get_holding_candles(
        "5m",
        BASE_HOLDING_CANDLES,
    )

    five_minute_available = len(
        archives["5m"]
    )

    print(
        "5m candles required for a full base holding period: "
        f"{five_minute_required}"
    )

    print(
        f"5m candles currently available: "
        f"{five_minute_available}"
    )

    if five_minute_available <= five_minute_required:
        print(
            "WARNING: The current 5m archive is too short "
            "to provide a full base holding period."
        )

    print(
        "\nDaily direction uses only completed daily candles."
    )

    print(
        "Only 4H BOS/CHOCH events create trade setups."
    )

    print(
        "Lower timeframes are used only for entry confirmation."
    )

    print(
        "Each 4H setup can produce at most one trade."
    )

    print(
        "Each lower-timeframe confirmation event can be consumed once."
    )

    print(
        "Stops are based on confirmed 4H opposite swings."
    )

    print(
        "Holding periods are measured in equivalent 4H candles."
    )

    print(
        "Same-candle stop and target conflicts use stop-first logic."
    )

    print(
        "No live orders are submitted."
    )

    print(
        "\nHIERARCHICAL BACKTEST COMPLETED"
    )

    return {
        "setups": setups,
        "baseline_trades": baseline_trades,
        "managed_trades": managed_trades,
        "extensions": extensions,
        "statistics": statistics,
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
        f"Base holding period: "
        f"{BASE_HOLDING_CANDLES} equivalent 4H candles"
    )
    print(
        f"Setup expiry: "
        f"{SETUP_MAX_WAIT_4H_CANDLES} equivalent 4H candles"
    )
    print(
        f"Profit lock trigger: "
        f"+{PROFIT_LOCK_TRIGGER_R}R"
    )
    print(
        f"Extended horizons: {EXTENDED_HORIZONS}"
    )
    print(
        f"Fee per side: {FEE_RATE * 100:.3f}%"
    )
    print(
        f"Slippage per side: {SLIPPAGE_RATE * 100:.3f}%"
    )
    print("Entry price: Next lower-timeframe candle Open")
    print("Same-candle SL/TP: Stop Loss First")
    print("Profit lock activation: Next candle after trigger")
    print("Trading execution: DISABLED")
    print("Archive modification: NO")

    detect_market_structure = load_structure_module()

    analyze_hierarchical_strategy(
        detect_market_structure
    )


if __name__ == "__main__":
    main()
