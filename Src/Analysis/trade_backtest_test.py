
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
PROFIT_LOCK_TRIGGER_R = 0.5

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

        # Activate a previously triggered profit lock.
        # It becomes active only from the next candle.
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

        # Detect +0.5R only after checking existing SL/TP.
        # The new stop is activated from the next candle.
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

    if len(structure) != len(data):
        raise ValueError(
            f"{timeframe}: Structure output length mismatch"
        )

    required_structure_columns = set()

    for config in STRUCTURE_CONFIG.values():
        required_structure_columns.add(config["event_column"])
        required_structure_columns.add(config["swing_high_column"])
        required_structure_columns.add(config["swing_low_column"])

    missing_structure = required_structure_columns - set(structure.columns)

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
        "24-candle horizon are included."
    )
    print(
        "Extended simulations restart from the original entry "
        "with the original stop and target."
    )


def print_strategy_comparison(baseline_trades, managed_trades):
    baseline = pd.DataFrame(baseline_trades)
    managed = pd.DataFrame(managed_trades)

    print("\n" + "=" * 70)
    print("BASELINE VS PROFIT LOCK COMPARISON")
    print("=" * 70)

    baseline_summary = calculate_summary(baseline_trades)
    managed_summary = calculate_summary(managed_trades)

    print("\nBASELINE:")
    print(baseline_summary)

    print("\nPROFIT LOCK:")
    print(managed_summary)

    baseline_time_exits = baseline_summary["time_exits"]
    managed_time_exits = managed_summary["time_exits"]

    baseline_count = baseline_summary["trades"]
    managed_count = managed_summary["trades"]

    baseline_pct = (
        baseline_time_exits / baseline_count * 100
        if baseline_count else 0
    )
    managed_pct = (
        managed_time_exits / managed_count * 100
        if managed_count else 0
    )

    reduction_count = baseline_time_exits - managed_time_exits
    reduction_pct = (
        reduction_count / baseline_time_exits * 100
        if baseline_time_exits else 0
    )

    print("\nTIME EXIT COMPARISON:")
    print(f"Baseline TIME_EXIT count: {baseline_time_exits}")
    print(f"Profit Lock TIME_EXIT count: {managed_time_exits}")
    print(f"Baseline TIME_EXIT rate: {baseline_pct:.2f}%")
    print(f"Profit Lock TIME_EXIT rate: {managed_pct:.2f}%")
    print(f"TIME_EXIT count reduction: {reduction_count}")
    print(f"TIME_EXIT count reduction percentage: {reduction_pct:.2f}%")

    print("\nNET PERFORMANCE CHANGE:")
    print(
        "Average Net R change: "
        f"{managed_summary['avg_net_R'] - baseline_summary['avg_net_R']:.4f}"
    )

    print(
        "Profit Factor change: "
        f"{baseline_summary['profit_factor']} -> "
        f"{managed_summary['profit_factor']}"
    )

    print("\nPROFIT LOCK ACTIVATION:")
    activated = int(managed["profit_lock_activated"].sum())
    lock_exits = managed_summary["profit_lock_exits"]

    print(f"Trades where +0.5R was reached: {activated}")
    print(f"Trades closed by profit lock: {lock_exits}")
    print(
        "Trades reaching +0.5R but not closed by profit lock: "
        f"{activated - lock_exits}"
    )

    print("\nPROFIT LOCK RESULTS BY TIMEFRAME:")
    print_grouped_summary(
        managed,
        ["timeframe"],
        "PROFIT LOCK BY TIMEFRAME",
    )

    print("\nBASELINE RESULTS BY TIMEFRAME:")
    print_grouped_summary(
        baseline,
        ["timeframe"],
        "BASELINE BY TIMEFRAME",
    )

    print("\nPROFIT LOCK EXIT REASONS:")
    print(managed["exit_reason"].value_counts().to_dict())


def main():
    print("=" * 70)
    print("BTC MARKET STRUCTURE TRADE BACKTEST")
    print("BASELINE VS PROFIT LOCK")
    print("=" * 70)

    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Risk/Reward: 1:{RISK_REWARD}")
    print(f"Base holding candles: {BASE_HOLDING_CANDLES}")
    print(f"Profit lock trigger: +{PROFIT_LOCK_TRIGGER_R}R")
    print(f"Extended horizons: {EXTENDED_HORIZONS}")
    print(f"Fee per side: {FEE_RATE * 100:.3f}%")
    print(f"Slippage per side: {SLIPPAGE_RATE * 100:.3f}%")
    print("Entry: Next candle Open")
    print("Same-candle SL/TP: Stop Loss First")
    print("Profit lock activation: Next candle after +0.5R trigger")
    print("Signal sources: INTERNAL and EXTERNAL")
    print("Stops: Structure-specific confirmed opposite swings")
    print("Trading execution: DISABLED")
    print("Archive modification: NO")
    print("-" * 70)

    detect_market_structure = load_structure_module()

    all_baseline_trades = []
    all_managed_trades = []
    all_extensions = []
    successful_timeframes = 0

    for timeframe in TIMEFRAMES:
        try:
            baseline, managed, extensions, skipped = analyze_timeframe(
                timeframe,
                detect_market_structure,
            )

            all_baseline_trades.extend(baseline)
            all_managed_trades.extend(managed)
            all_extensions.extend(extensions)
            successful_timeframes += 1

            print(
                f"{timeframe}: PASS | "
                f"Baseline trades: {len(baseline)} | "
                f"Profit Lock trades: {len(managed)} | "
                f"Skipped: {sum(skipped.values())}"
            )

            for reason, count in sorted(skipped.items()):
                print(f"  {reason}: {count}")

            baseline_time_exits = sum(
                trade["exit_reason"] == "TIME_EXIT"
                for trade in baseline
            )

            managed_time_exits = sum(
                trade["exit_reason"] == "TIME_EXIT"
                for trade in managed
            )

            print(f"  Baseline TIME_EXIT: {baseline_time_exits}")
            print(f"  Profit Lock TIME_EXIT: {managed_time_exits}")

        except Exception as error:
            print(f"{timeframe}: FAIL | {error}")

    print("-" * 70)
    print(f"Successful timeframes: {successful_timeframes}")
    print(
        f"Failed timeframes: {len(TIMEFRAMES) - successful_timeframes}"
    )
    print(f"Total baseline trades: {len(all_baseline_trades)}")
    print(f"Total profit lock trades: {len(all_managed_trades)}")

    if not all_baseline_trades:
        print("No trades available for summary.")
        print("\nBACKTEST COMPLETED")
        return

    baseline_frame = pd.DataFrame(all_baseline_trades)
    managed_frame = pd.DataFrame(all_managed_trades)

    print("\nBASELINE OVERALL SUMMARY")
    print(calculate_summary(all_baseline_trades))

    print("\nPROFIT LOCK OVERALL SUMMARY")
    print(calculate_summary(all_managed_trades))

    print_strategy_comparison(
        all_baseline_trades,
        all_managed_trades,
    )

    print_grouped_summary(
        baseline_frame,
        ["timeframe", "structure_type", "signal_type"],
        "BASELINE BY TIMEFRAME, STRUCTURE AND SIGNAL",
    )

    print_grouped_summary(
        managed_frame,
        ["timeframe", "structure_type", "signal_type"],
        "PROFIT LOCK BY TIMEFRAME, STRUCTURE AND SIGNAL",
    )

    baseline_time_exit_frame = baseline_frame[
        baseline_frame["exit_reason"] == "TIME_EXIT"
    ]

    managed_time_exit_frame = managed_frame[
        managed_frame["exit_reason"] == "TIME_EXIT"
    ]

    print("\nBASELINE TIME EXIT DIAGNOSTICS")
    print(f"Total: {len(baseline_time_exit_frame)}")

    if not baseline_time_exit_frame.empty:
        print(
            calculate_summary(
                baseline_time_exit_frame.to_dict("records")
            )
        )

    print("\nPROFIT LOCK TIME EXIT DIAGNOSTICS")
    print(f"Total: {len(managed_time_exit_frame)}")

    if not managed_time_exit_frame.empty:
        print(
            calculate_summary(
                managed_time_exit_frame.to_dict("records")
            )
        )

        print("\nPROFIT LOCK TIME EXIT BY TIMEFRAME")
        print_grouped_summary(
            managed_time_exit_frame,
            ["timeframe"],
            "PROFIT LOCK TIME EXIT BY TIMEFRAME",
        )

    print_extension_report(all_extensions)

    print("\nIMPORTANT LIMITATIONS")
    print("- Baseline and Profit Lock use identical signals and initial stops.")
    print("- Each signal is simulated independently in both strategies.")
    print("- Profit lock activates from the candle after the +0.5R trigger.")
    print("- If +0.5R is first reached on the final holding candle, the trade remains a TIME_EXIT.")
    print("- OHLC data cannot determine exact intrabar price movement order.")
    print("- Same-candle stop and target conflicts are resolved conservatively.")
    print("- Fees and slippage are applied to every simulated trade.")
    print("- Extended holding analysis uses baseline TIME_EXIT trades only.")
    print("- No position sizing, portfolio simulation, or capital compounding is applied.")
    print("- Historical results do not guarantee future profitability.")

    print("\nBACKTEST COMPLETED")


if __name__ == "__main__":
    main()
