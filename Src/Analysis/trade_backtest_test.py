
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
MAX_HOLDING_CANDLES = 24
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


def simulate_trade(data, signal_index, direction, stop_price):
    entry_index = signal_index + 1

    if entry_index + MAX_HOLDING_CANDLES > len(data):
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

    end_index = entry_index + MAX_HOLDING_CANDLES

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

                trade, reason = simulate_trade(
                    data,
                    index,
                    direction,
                    stop_price,
                )

                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                trade["timeframe"] = timeframe
                trade["structure_type"] = structure_name
                trade["signal_type"] = event_name

                trades.append(trade)

    return trades, skipped


def main():
    print("=" * 70)
    print("BTC MARKET STRUCTURE TRADE BACKTEST")
    print("=" * 70)

    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Risk/Reward: 1:{RISK_REWARD}")
    print(f"Maximum holding candles: {MAX_HOLDING_CANDLES}")
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
    all_skipped = {}
    successful_timeframes = 0

    for timeframe in TIMEFRAMES:
        try:
            trades, skipped = analyze_timeframe(
                timeframe,
                detect_market_structure,
            )

            all_trades.extend(trades)
            successful_timeframes += 1

            for reason, count in skipped.items():
                all_skipped[reason] = (
                    all_skipped.get(reason, 0) + count
                )

            print(
                f"{timeframe}: PASS | "
                f"Executed trades: {len(trades)} | "
                f"Skipped: {sum(skipped.values())}"
            )

            for reason, count in sorted(skipped.items()):
                print(f"  {reason}: {count}")

        except Exception as error:
            print(f"{timeframe}: FAIL | {error}")

    print("-" * 70)
    print(
        f"Successful timeframes: {successful_timeframes}"
    )
    print(
        f"Failed timeframes: {len(TIMEFRAMES) - successful_timeframes}"
    )
    print(f"Total executed trades: {len(all_trades)}")

    if not all_trades:
        print("No trades available for summary.")
        print("\nBACKTEST COMPLETED")
        return

    frame = pd.DataFrame(all_trades)

    print("\nSUMMARY BY TIMEFRAME")

    for timeframe in TIMEFRAMES:
        subset = frame[
            frame["timeframe"] == timeframe
        ]

        if subset.empty:
            continue

        summary = calculate_summary(
            subset.to_dict("records")
        )

        print(f"{timeframe}: {summary}")

    print("\nSUMMARY BY TIMEFRAME, STRUCTURE AND SIGNAL")

    grouped = frame.groupby(
        ["timeframe", "structure_type", "signal_type"]
    )

    for (timeframe, structure_type, signal_type), group in grouped:
        summary = calculate_summary(
            group.to_dict("records")
        )

        print(
            f"{timeframe} | {structure_type} | "
            f"{signal_type}: {summary}"
        )

    print("\nSUMMARY BY STRUCTURE")

    for structure_type, group in frame.groupby("structure_type"):
        summary = calculate_summary(
            group.to_dict("records")
        )

        print(f"{structure_type}: {summary}")

    print("\nOVERALL SUMMARY")
    print(calculate_summary(all_trades))

    print("\nEXIT REASONS")
    print(
        frame["exit_reason"].value_counts().to_dict()
    )

    print("\nIMPORTANT LIMITATIONS")
    print("- Signals may overlap; this is not a portfolio simulation.")
    print("- Internal and external signals are evaluated independently.")
    print("- Each trade is evaluated independently.")
    print("- Stops use the last confirmed opposite swing of the same structure.")
    print("- No position sizing or capital compounding is applied.")
    print("- Results are historical analysis, not a profitability guarantee.")

    print("\nBACKTEST COMPLETED")


if __name__ == "__main__":
    main()
