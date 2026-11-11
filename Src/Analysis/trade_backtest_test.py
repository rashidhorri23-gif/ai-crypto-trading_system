
from pathlib import Path
import sys
import math
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]

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

FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002


def load_structure_module():
    sys.path.insert(0, str(PROJECT_ROOT))
    from Src.Analysis.market_structure import detect_market_structure
    return detect_market_structure


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


def calculate_excursions(data, entry_index, exit_index, direction, entry_price, risk_per_unit):
    trade_candles = data.iloc[
        entry_index:exit_index + 1
    ]

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
):
    entry_index = signal_index + 1

    if entry_index + max_holding_candles > len(data):
        return None, "INSUFFICIENT_FUTURE_DATA"

    entry_price = float(data.iloc[entry_index]["open"])
    stop_price = float(stop_price)

    if not math.isfinite(entry_price) or not math.isfinite(stop_price):
        return None, "INVALID_PRICE"

    if entry_price <= 0 or stop_price <= 0:
        return None, "INVALID_PRICE"

    if direction == 1 and stop_price >= entry_price:
        return None, "INVALID_LONG_STOP"

    if direction == -1 and stop_price <= entry_price:
        return None, "INVALID_SHORT_STOP"

    risk_per_unit = abs(entry_price - stop_price)

    if risk_per_unit <= 0:
        return None, "ZERO_RISK"

    if direction == 1:
        target_price = entry_price + RISK_REWARD * risk_per_unit
    else:
        target_price = entry_price - RISK_REWARD * risk_per_unit

    exit_price = None
    exit_reason = None
    exit_index = None

    end_index = entry_index + max_holding_candles

    for current_index in range(entry_index, end_index):
        candle = data.iloc[current_index]

        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])

        if direction == 1:
            if open_price <= stop_price:
                exit_price = open_price
                exit_reason = "STOP_GAP"

            elif open_price >= target_price:
                exit_price = open_price
                exit_reason = "TARGET_GAP"

            else:
                stop_hit = low_price <= stop_price
                target_hit = high_price >= target_price

                if stop_hit:
                    exit_price = stop_price
                    exit_reason = (
                        "STOP_BOTH_HIT" if target_hit else "STOP"
                    )

                elif target_hit:
                    exit_price = target_price
                    exit_reason = "TARGET"

        else:
            if open_price >= stop_price:
                exit_price = open_price
                exit_reason = "STOP_GAP"

            elif open_price <= target_price:
                exit_price = open_price
                exit_reason = "TARGET_GAP"

            else:
                stop_hit = high_price >= stop_price
                target_hit = low_price <= target_price

                if stop_hit:
                    exit_price = stop_price
                    exit_reason = (
                        "STOP_BOTH_HIT" if target_hit else "STOP"
                    )

                elif target_hit:
                    exit_price = target_price
                    exit_reason = "TARGET"

        if exit_price is not None:
            exit_index = current_index
            break

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
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_R": gross_r,
        "net_R": net_r,
        "net_return_pct": net_return_pct,
        "risk_per_unit": risk_per_unit,
        "holding_candles": exit_index - entry_index + 1,
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
        "time_exits": int(
            (frame["exit_reason"] == "TIME_EXIT").sum()
        ),
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

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

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
            errors="raise"
        )

    structure = detect_market_structure(
        data,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
    )

    if len(structure) != len(data):
        raise ValueError(
            f"{timeframe}: Structure output length mismatch"
        )

    required_structure_columns = set()

    for config in STRUCTURE_CONFIG.values():
        required_structure_columns.add(
            config["event_column"]
        )
        required_structure_columns.add(
            config["swing_high_column"]
        )
        required_structure_columns.add(
            config["swing_low_column"]
        )

    missing_structure = (
        required_structure_columns - set(structure.columns)
    )

    if missing_structure:
        raise ValueError(
            f"{timeframe}: Missing structure columns: "
            f"{sorted(missing_structure)}"
        )

    latest_swings = {}

    for structure_name, config in STRUCTURE_CONFIG.items():
        latest_swings[structure_name] = {
            "high": structure[config["swing_high_column"]].ffill(),
            "low": structure[config["swing_low_column"]].ffill(),
        }

    trades = []
    extensions = []
    skipped = {}

    for index in range(len(data)):
        for structure_name, config in STRUCTURE_CONFIG.items():
            event_value = structure.iloc[index][
                config["event_column"]
            ]

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
                )

                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                base_trade["timeframe"] = timeframe
                base_trade["structure_type"] = structure_name
                base_trade["signal_type"] = event_name

                trades.append(base_trade)

                if base_trade["exit_reason"] != "TIME_EXIT":
                    continue

                for horizon in EXTENDED_HORIZONS:
                    extended_trade, extension_reason = simulate_trade(
                        data,
                        index,
                        direction,
                        stop_price,
                        horizon,
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

    return trades, extensions, skipped


def print_grouped_summary(frame, group_columns, heading):
    print(f"\n{heading}")

    grouped = frame.groupby(group_columns)

    for group_key, group in grouped:
        summary = calculate_summary(
            group.to_dict("records")
        )

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        label = " | ".join(str(value) for value in group_key)
        print(f"{label}: {summary}")


def print_extension_report(extensions):
    print("\nEXTENDED HOLDING ANALYSIS")

    if not extensions:
        print("No Time Exit trades available for extension analysis.")
        return

    extension_frame = pd.DataFrame(extensions)

    for horizon in EXTENDED_HORIZONS:
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
        "Only trades that reached TIME_EXIT at the base "
        "24-candle horizon are included."
    )
    print(
        "Extended simulations restart from the original entry "
        "with the original stop and target."
    )


def main():
    print("=" * 70)
    print("BTC MARKET STRUCTURE TRADE BACKTEST")
    print("=" * 70)

    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Risk/Reward: 1:{RISK_REWARD}")
    print(f"Base holding candles: {BASE_HOLDING_CANDLES}")
    print(f"Extended horizons: {EXTENDED_HORIZONS}")
    print(f"Fee per side: {FEE_RATE * 100:.3f}%")
    print(f"Slippage per side: {SLIPPAGE_RATE * 100:.3f}%")
    print("Entry: Next candle Open")
    print("Same-candle SL/TP: Stop Loss First")
    print("Signal sources: INTERNAL and EXTERNAL")
    print("Stops: Structure-specific confirmed opposite swings")
    print("Trading execution: DISABLED")
    print("Archive modification: NO")
    print("-" * 70)

    detect_market_structure = load_structure_module()

    all_trades = []
    all_extensions = []
    successful_timeframes = 0

    for timeframe in TIMEFRAMES:
        try:
            trades, extensions, skipped = analyze_timeframe(
                timeframe,
                detect_market_structure,
            )

            all_trades.extend(trades)
            all_extensions.extend(extensions)
            successful_timeframes += 1

            print(
                f"{timeframe}: PASS | "
                f"Executed trades: {len(trades)} | "
                f"Skipped: {sum(skipped.values())}"
            )

            for reason, count in sorted(skipped.items()):
                print(f"  {reason}: {count}")

            time_exit_count = sum(
                1 for trade in trades
                if trade["exit_reason"] == "TIME_EXIT"
            )

            print(f"  Base TIME_EXIT trades: {time_exit_count}")

        except Exception as error:
            print(f"{timeframe}: FAIL | {error}")

    print("-" * 70)
    print(f"Successful timeframes: {successful_timeframes}")
    print(
        f"Failed timeframes: {len(TIMEFRAMES) - successful_timeframes}"
    )
    print(f"Total executed trades: {len(all_trades)}")

    if not all_trades:
        print("No trades available for summary.")
        print("\nBACKTEST COMPLETED")
        return

    frame = pd.DataFrame(all_trades)

    print_grouped_summary(
        frame,
        ["timeframe"],
        "SUMMARY BY TIMEFRAME",
    )

    print_grouped_summary(
        frame,
        ["timeframe", "structure_type", "signal_type"],
        "SUMMARY BY TIMEFRAME, STRUCTURE AND SIGNAL",
    )

    print_grouped_summary(
        frame,
        ["structure_type"],
        "SUMMARY BY STRUCTURE",
    )

    print("\nOVERALL SUMMARY")
    print(calculate_summary(all_trades))

    print("\nEXIT REASONS")
    print(frame["exit_reason"].value_counts().to_dict())

    print("\nTIME EXIT DIAGNOSTICS")
    time_exit_frame = frame[
        frame["exit_reason"] == "TIME_EXIT"
    ]

    print(f"Total TIME_EXIT trades: {len(time_exit_frame)}")

    if not time_exit_frame.empty:
        print("TIME_EXIT performance:")
        print(
            calculate_summary(
                time_exit_frame.to_dict("records")
            )
        )

        print_grouped_summary(
            time_exit_frame,
            ["timeframe"],
            "TIME EXIT BY TIMEFRAME",
        )

        print_grouped_summary(
            time_exit_frame,
            ["timeframe", "structure_type", "signal_type"],
            "TIME EXIT BY TIMEFRAME, STRUCTURE AND SIGNAL",
        )

        print("\nTIME EXIT NET R DISTRIBUTION")
        print(
            time_exit_frame["net_R"].describe().round(4).to_dict()
        )

        print("\nTIME EXIT MFE / MAE")
        print({
            "average_mfe_R": round(
                time_exit_frame["mfe_R"].mean(), 4
            ),
            "average_mae_R": round(
                time_exit_frame["mae_R"].mean(), 4
            ),
            "median_mfe_R": round(
                time_exit_frame["mfe_R"].median(), 4
            ),
            "median_mae_R": round(
                time_exit_frame["mae_R"].median(), 4
            ),
        })

    print_extension_report(all_extensions)

    print("\nIMPORTANT LIMITATIONS")
    print("- Signals may overlap; this is not a portfolio simulation.")
    print("- Internal and external signals are evaluated independently.")
    print("- Each trade is evaluated independently.")
    print("- Stops use the last confirmed opposite swing of the same structure.")
    print("- Extended tests include only trades that reached the base TIME_EXIT.")
    print("- Extended tests retain the original stop and target.")
    print("- MFE and MAE use candle high/low, including the exit candle.")
    print("- OHLC data cannot determine the exact intrabar order of all price movements.")
    print("- No position sizing or capital compounding is applied.")
    print("- Results are historical analysis, not a profitability guarantee.")

    print("\nBACKTEST COMPLETED")


if __name__ == "__main__":
    main()
