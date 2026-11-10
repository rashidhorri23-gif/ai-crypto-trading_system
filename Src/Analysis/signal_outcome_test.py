
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.Analysis.market_structure import detect_market_structure


TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]
HORIZONS = [1, 3, 6, 12, 24]

EVENT_TYPES = [
    "BOS_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BULLISH",
    "CHOCH_BEARISH",
]


def load_archive(timeframe):
    file_path = ARCHIVE_DIR / f"BTC_USDT_SPOT_{timeframe}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Archive not found: {file_path}")

    data = pd.read_csv(file_path)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {timeframe}: {missing}"
        )

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[required].isna().any().any():
        raise ValueError(f"Null values found in {timeframe}")

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {timeframe}")

    return data


def calculate_outcome(data, signal_index, direction, horizon):
    entry_index = signal_index + 1

    if entry_index >= len(data):
        return None

    end_index = entry_index + horizon

    if end_index > len(data):
        return None

    entry_price = float(data.iloc[entry_index]["open"])

    if entry_price <= 0:
        raise ValueError("Invalid entry price")

    future = data.iloc[entry_index:end_index]

    final_close = float(future.iloc[-1]["close"])
    highest_price = float(future["high"].max())
    lowest_price = float(future["low"].min())

    if direction == 1:
        directional_return = (
            (final_close - entry_price) / entry_price
        ) * 100

        mfe = (
            (highest_price - entry_price) / entry_price
        ) * 100

        mae = (
            (lowest_price - entry_price) / entry_price
        ) * 100

    else:
        directional_return = (
            (entry_price - final_close) / entry_price
        ) * 100

        mfe = (
            (entry_price - lowest_price) / entry_price
        ) * 100

        mae = (
            (entry_price - highest_price) / entry_price
        ) * 100

    return {
        "return": directional_return,
        "mfe": mfe,
        "mae": mae,
        "win": directional_return > 0,
    }


def evaluate_event_type(data, event_indices, direction, horizon):
    outcomes = []

    for signal_index in event_indices:
        result = calculate_outcome(
            data,
            signal_index,
            direction,
            horizon,
        )

        if result is not None:
            outcomes.append(result)

    if not outcomes:
        return None

    results = pd.DataFrame(outcomes)

    return {
        "observations": len(results),
        "win_rate": results["win"].mean() * 100,
        "average_return": results["return"].mean(),
        "median_return": results["return"].median(),
        "average_mfe": results["mfe"].mean(),
        "average_mae": results["mae"].mean(),
    }


def analyze_timeframe(timeframe):
    data = load_archive(timeframe)

    structure = detect_market_structure(
        data,
        left_bars=3,
        right_bars=3,
    )

    if len(structure) != len(data):
        raise ValueError(
            f"Structure row count mismatch in {timeframe}"
        )

    print()
    print("=" * 65)
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(data)}")
    print("=" * 65)

    event_counts = {}

    for event_type in EVENT_TYPES:
        indices = structure.index[
            structure["structure_event"] == event_type
        ].tolist()

        direction = 1 if event_type.endswith("BULLISH") else -1

        event_counts[event_type] = len(indices)

        print()
        print(f"EVENT TYPE: {event_type}")
        print(f"Signal Count: {len(indices)}")

        for horizon in HORIZONS:
            result = evaluate_event_type(
                data,
                indices,
                direction,
                horizon,
            )

            if result is None:
                print(
                    f"Horizon {horizon}: "
                    "No eligible observations"
                )
                continue

            print(
                f"Horizon {horizon}: "
                f"N={result['observations']}, "
                f"WinRate={result['win_rate']:.2f}%, "
                f"AvgReturn={result['average_return']:.5f}%, "
                f"MedianReturn={result['median_return']:.5f}%, "
                f"AvgMFE={result['average_mfe']:.5f}%, "
                f"AvgMAE={result['average_mae']:.5f}%"
            )

    total_signals = sum(event_counts.values())

    print()
    print(f"TOTAL SIGNALS: {total_signals}")
    print(f"TIMEFRAME STATUS: PASS")

    return {
        "timeframe": timeframe,
        "candles": len(data),
        "signals": total_signals,
    }


def main():
    print("BOS AND CHOCH SIGNAL OUTCOME TEST")
    print("=" * 65)
    print("Entry: Next Candle Open")
    print("Outcome: Subsequent Candle Close")
    print(f"Forward Horizons: {HORIZONS}")
    print("Market: BTC-USDT SPOT")
    print("Data Source: Local OKX Archive")
    print("Execution: Read Only")
    print("Trading: DISABLED")
    print("Fees and Slippage: NOT INCLUDED")
    print("Cross-Timeframe Aggregation: DISABLED")

    successful = []
    failed = []

    for timeframe in TIMEFRAMES:
        try:
            result = analyze_timeframe(timeframe)
            successful.append(result)

        except Exception as error:
            failed.append({
                "timeframe": timeframe,
                "error": str(error),
            })

            print()
            print(f"TIMEFRAME FAILED: {timeframe}")
            print(f"ERROR: {error}")

    print()
    print("=" * 65)
    print("FINAL EXECUTION REPORT")
    print("=" * 65)
    print(f"Successful Timeframes: {len(successful)}")
    print(f"Failed Timeframes: {len(failed)}")
    print(f"Total Timeframes: {len(TIMEFRAMES)}")

    if failed:
        print("Failed List:")
        for item in failed:
            print(f"{item['timeframe']}: {item['error']}")

    print("ARCHIVE MODIFICATION: NO")
    print("TRADING EXECUTION: DISABLED")

    if failed:
        raise RuntimeError(
            "Signal outcome analysis failed for one or more timeframes"
        )


if __name__ == "__main__":
    main()
