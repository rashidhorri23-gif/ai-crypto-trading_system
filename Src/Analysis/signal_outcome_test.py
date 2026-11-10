
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_structure import detect_market_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]
HORIZONS = [1, 3, 6, 12, 24]

LEFT_BARS = 3
RIGHT_BARS = 3


def evaluate_event(result, event_index, horizon):
    entry = float(result.iloc[event_index]["close"])
    event = str(result.iloc[event_index]["structure_event"])

    direction = 1 if event.endswith("BULLISH") else -1

    future = result.iloc[
        event_index + 1:event_index + horizon + 1
    ]

    if len(future) < horizon:
        return None

    final_close = float(future.iloc[-1]["close"])

    directional_return = (
        (final_close / entry) - 1
    ) * direction * 100

    future_high = future["high"].astype(float).max()
    future_low = future["low"].astype(float).min()

    if direction == 1:
        favorable = ((future_high / entry) - 1) * 100
        adverse = ((future_low / entry) - 1) * 100
    else:
        favorable = ((entry - future_low) / entry) * 100
        adverse = ((entry - future_high) / entry) * 100

    return {
        "return_pct": directional_return,
        "favorable_pct": favorable,
        "adverse_pct": adverse,
    }


def evaluate_timeframe(timeframe):
    filename = f"BTC_USDT_SPOT_{timeframe}.csv"
    file_path = ARCHIVE_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Archive not found: {filename}")

    candles = pd.read_csv(file_path)

    if candles.empty:
        raise ValueError(f"Archive is empty: {filename}")

    result = detect_market_structure(
        candles,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "structure_event",
    ]

    missing = [
        column for column in required
        if column not in result.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result = result.reset_index(drop=True)

    events = []

    for index, row in result.iterrows():
        event = str(row["structure_event"])

        if event not in {
            "BOS_BULLISH",
            "BOS_BEARISH",
            "CHOCH_BULLISH",
            "CHOCH_BEARISH",
        }:
            continue

        events.append({
            "index": index,
            "event": event,
        })

    print("")
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(result)}")
    print(f"Total Signals: {len(events)}")

    timeframe_reports = []

    for event_type in [
        "BOS_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BULLISH",
        "CHOCH_BEARISH",
    ]:
        selected = [
            item for item in events
            if item["event"] == event_type
        ]

        print("")
        print(f"EVENT TYPE: {event_type}")
        print(f"Signal Count: {len(selected)}")

        for horizon in HORIZONS:
            outcomes = []

            for item in selected:
                outcome = evaluate_event(
                    result,
                    item["index"],
                    horizon,
                )

                if outcome is not None:
                    outcomes.append(outcome)

            if not outcomes:
                print(
                    f"Horizon {horizon}: "
                    "No complete observations"
                )
                continue

            returns = np.array([
                item["return_pct"] for item in outcomes
            ])

            favorable = np.array([
                item["favorable_pct"] for item in outcomes
            ])

            adverse = np.array([
                item["adverse_pct"] for item in outcomes
            ])

            wins = int(np.sum(returns > 0))
            losses = int(np.sum(returns < 0))
            flat = int(np.sum(returns == 0))

            win_rate = wins / len(returns) * 100

            avg_return = float(np.mean(returns))
            median_return = float(np.median(returns))
            avg_favorable = float(np.mean(favorable))
            avg_adverse = float(np.mean(adverse))

            report = {
                "timeframe": timeframe,
                "event": event_type,
                "horizon": horizon,
                "observations": len(returns),
                "wins": wins,
                "losses": losses,
                "flat": flat,
                "win_rate_pct": win_rate,
                "average_return_pct": avg_return,
                "median_return_pct": median_return,
                "average_favorable_pct": avg_favorable,
                "average_adverse_pct": avg_adverse,
            }

            timeframe_reports.append(report)

            print(
                f"Horizon {horizon}: "
                f"N={len(returns)}, "
                f"WinRate={win_rate:.2f}%, "
                f"AvgReturn={avg_return:.5f}%, "
                f"MedianReturn={median_return:.5f}%, "
                f"AvgMFE={avg_favorable:.5f}%, "
                f"AvgMAE={avg_adverse:.5f}%"
            )

    return timeframe_reports


def main():
    print("BOS AND CHOCH SIGNAL OUTCOME TEST")
    print("================================")
    print(f"Left Bars: {LEFT_BARS}")
    print(f"Right Bars: {RIGHT_BARS}")
    print(f"Forward Horizons: {HORIZONS}")
    print("Market: BTC-USDT SPOT")
    print("Data Source: Local OKX Archive")
    print("Execution: Read Only")
    print("Trading: DISABLED")
    print("Fees and Slippage: NOT INCLUDED")

    all_reports = []
    failed = []

    for timeframe in TIMEFRAMES:
        try:
            reports = evaluate_timeframe(timeframe)
            all_reports.extend(reports)

        except Exception as error:
            print("")
            print(f"TIMEFRAME ERROR: {timeframe}")
            print(f"ERROR DETAILS: {error}")
            failed.append(timeframe)

    print("")
    print("FINAL SIGNAL OUTCOME SUMMARY")
    print("============================")

    if all_reports:
        summary = pd.DataFrame(all_reports)

        for event_type in [
            "BOS_BULLISH",
            "BOS_BEARISH",
            "CHOCH_BULLISH",
            "CHOCH_BEARISH",
        ]:
            selected = summary[
                summary["event"] == event_type
            ]

            if selected.empty:
                continue

            print("")
            print(f"EVENT: {event_type}")

            for horizon in HORIZONS:
                rows = selected[
                    selected["horizon"] == horizon
                ]

                if rows.empty:
                    continue

                total = int(rows["observations"].sum())

                weighted_wins = int(rows["wins"].sum())

                total_win_rate = (
                    weighted_wins / total * 100
                    if total else 0
                )

                weighted_return = float(
                    np.average(
                        rows["average_return_pct"],
                        weights=rows["observations"],
                    )
                )

                print(
                    f"Horizon {horizon}: "
                    f"Observations={total}, "
                    f"WinRate={total_win_rate:.2f}%, "
                    f"WeightedAvgReturn={weighted_return:.5f}%"
                )

    print("")
    print("FINAL EXECUTION REPORT")
    print("======================")
    print(f"Successful Timeframes: {len(TIMEFRAMES) - len(failed)}")
    print(f"Failed Timeframes: {len(failed)}")
    print(f"Total Timeframes: {len(TIMEFRAMES)}")
    print(f"Failed List: {failed}")
    print("ARCHIVE MODIFICATION: NO")
    print("TRADING EXECUTION: DISABLED")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
