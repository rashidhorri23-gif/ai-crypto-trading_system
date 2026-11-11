
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.Analysis.daily_structure import analyze_daily_structure


TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]
HORIZONS = [1, 3, 6, 12, 24]

EVENT_TYPES = [
    "BOS_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BULLISH",
    "CHOCH_BEARISH",
]

DIRECTIONS = {
    "LONG": 1,
    "SHORT": -1,
}


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

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {timeframe}: {missing}"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise"
        )

    if data[required].isna().any().any():
        raise ValueError(
            f"Null values found in {timeframe}"
        )

    if not data[required].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all().all():
        raise ValueError(
            f"Non-finite values found in {timeframe}"
        )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps in {timeframe}"
        )

    if (data["timestamp"] <= 0).any():
        raise ValueError(
            f"Invalid timestamps in {timeframe}"
        )

    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(
            f"Non-positive prices in {timeframe}"
        )

    if (data["volume"] < 0).any():
        raise ValueError(
            f"Negative volume in {timeframe}"
        )

    invalid_high = data["high"] < data[
        ["open", "close", "low"]
    ].max(axis=1)

    invalid_low = data["low"] > data[
        ["open", "close", "high"]
    ].min(axis=1)

    if invalid_high.any() or invalid_low.any():
        raise ValueError(
            f"Invalid OHLC values in {timeframe}"
        )

    return data


def calculate_outcome(
    data,
    signal_index,
    direction,
    horizon
):
    if direction not in (1, -1):
        raise ValueError("Direction must be 1 or -1.")

    if horizon < 1:
        raise ValueError("Horizon must be positive.")

    entry_index = signal_index + 1

    if entry_index >= len(data):
        return None

    end_index = entry_index + horizon

    if end_index > len(data):
        return None

    entry_price = float(
        data.iloc[entry_index]["open"]
    )

    if entry_price <= 0:
        raise ValueError("Invalid entry price.")

    future = data.iloc[entry_index:end_index]

    if len(future) != horizon:
        return None

    final_close = float(
        future.iloc[-1]["close"]
    )

    highest_price = float(
        future["high"].max()
    )

    lowest_price = float(
        future["low"].min()
    )

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
        "entry_index": entry_index,
        "entry_timestamp": int(
            data.iloc[entry_index]["timestamp"]
        ),
        "entry_price": entry_price,
        "exit_index": end_index - 1,
        "exit_timestamp": int(
            future.iloc[-1]["timestamp"]
        ),
        "exit_price": final_close,
        "return": directional_return,
        "mfe": mfe,
        "mae": mae,
        "win": directional_return > 0,
    }


def evaluate_event_type(
    data,
    events,
    direction,
    horizon
):
    outcomes = []

    for event in events:
        signal_index = int(event["index"])

        if signal_index < 0 or signal_index >= len(data):
            continue

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


def collect_structure_events(structure_result):
    collected = []

    structure_groups = [
        ("INTERNAL", structure_result["internal_structure"]),
        ("EXTERNAL", structure_result["external_structure"]),
    ]

    for structure_type, structure_data in structure_groups:

        for event in structure_data["events"]:

            event_type = event.get("event")

            if event_type not in EVENT_TYPES:
                continue

            event_copy = dict(event)
            event_copy["structure_type"] = structure_type

            collected.append(event_copy)

    collected.sort(
        key=lambda item: (
            item["index"],
            item["structure_type"],
            item["event"],
        )
    )

    return collected


def print_result(
    structure_type,
    event_type,
    direction_name,
    horizon,
    result
):
    if result is None:
        print(
            f"  {direction_name} | Horizon {horizon}: "
            "No eligible observations"
        )
        return

    print(
        f"  {direction_name} | Horizon {horizon}: "
        f"N={result['observations']}, "
        f"WinRate={result['win_rate']:.2f}%, "
        f"AvgReturn={result['average_return']:.5f}%, "
        f"MedianReturn={result['median_return']:.5f}%, "
        f"AvgMFE={result['average_mfe']:.5f}%, "
        f"AvgMAE={result['average_mae']:.5f}%"
    )


def analyze_timeframe(timeframe):
    data = load_archive(timeframe)

    candles = data.to_dict(
        orient="records"
    )

    structure_result = analyze_daily_structure(
        candles,
        timeframe=timeframe
    )

    events = collect_structure_events(
        structure_result
    )

    print()
    print("=" * 75)
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(data)}")
    print(
        "Internal Trend: "
        f"{structure_result['internal_structure']['trend']}"
    )
    print(
        "Internal Event Bias: "
        f"{structure_result['internal_structure']['event_bias']}"
    )
    print(
        "External Trend: "
        f"{structure_result['external_structure']['trend']}"
    )
    print(
        "External Event Bias: "
        f"{structure_result['external_structure']['event_bias']}"
    )
    print("=" * 75)

    event_counts = {}

    for structure_type in ("INTERNAL", "EXTERNAL"):

        structure_events = [
            event for event in events
            if event["structure_type"] == structure_type
        ]

        print()
        print("-" * 75)
        print(f"STRUCTURE TYPE: {structure_type}")
        print("-" * 75)

        for event_type in EVENT_TYPES:

            matching_events = [
                event for event in structure_events
                if event["event"] == event_type
            ]

            event_count = len(matching_events)

            event_counts[
                (structure_type, event_type)
            ] = event_count

            print()
            print(f"EVENT TYPE: {event_type}")
            print(f"Signal Count: {event_count}")

            for direction_name, direction in DIRECTIONS.items():

                print(f"\nEvaluation Direction: {direction_name}")

                for horizon in HORIZONS:

                    result = evaluate_event_type(
                        data,
                        matching_events,
                        direction,
                        horizon,
                    )

                    print_result(
                        structure_type,
                        event_type,
                        direction_name,
                        horizon,
                        result,
                    )

    total_signals = sum(event_counts.values())

    internal_count = sum(
        count
        for (structure_type, _), count in event_counts.items()
        if structure_type == "INTERNAL"
    )

    external_count = sum(
        count
        for (structure_type, _), count in event_counts.items()
        if structure_type == "EXTERNAL"
    )

    print()
    print("=" * 75)
    print(f"TOTAL INTERNAL SIGNALS: {internal_count}")
    print(f"TOTAL EXTERNAL SIGNALS: {external_count}")
    print(f"TOTAL STRUCTURE EVENTS: {total_signals}")
    print("TIMEFRAME STATUS: PASS")
    print("=" * 75)

    return {
        "timeframe": timeframe,
        "candles": len(data),
        "internal_signals": internal_count,
        "external_signals": external_count,
        "total_signals": total_signals,
    }


def main():
    print("BOS AND CHOCH SIGNAL OUTCOME TEST")
    print("=" * 75)
    print("Structure Engine: DAILY_STRUCTURE")
    print("Entry: Next Candle Open")
    print("Outcome: Forward Candle Close")
    print(f"Forward Horizons: {HORIZONS}")
    print("Directions: LONG AND SHORT")
    print("Structure Types: INTERNAL AND EXTERNAL")
    print("Market: BTC-USDT SPOT")
    print("Data Source: Local OKX Archive")
    print("Execution: Read Only")
    print("Trading: DISABLED")
    print("Fees and Slippage: NOT INCLUDED")
    print("Stop Loss and Take Profit: NOT INCLUDED")
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
    print("=" * 75)
    print("FINAL EXECUTION REPORT")
    print("=" * 75)
    print(f"Successful Timeframes: {len(successful)}")
    print(f"Failed Timeframes: {len(failed)}")
    print(f"Total Timeframes: {len(TIMEFRAMES)}")

    total_internal = sum(
        item["internal_signals"]
        for item in successful
    )

    total_external = sum(
        item["external_signals"]
        for item in successful
    )

    total_signals = sum(
        item["total_signals"]
        for item in successful
    )

    print(f"Total Internal Signals: {total_internal}")
    print(f"Total External Signals: {total_external}")
    print(f"Total Structure Events: {total_signals}")

    if failed:
        print()
        print("FAILED TIMEFRAMES:")

        for item in failed:
            print(
                f"{item['timeframe']}: {item['error']}"
            )

    print()
    print("ARCHIVE MODIFICATION: NO")
    print("TRADING EXECUTION: DISABLED")

    if failed:
        raise RuntimeError(
            "Signal outcome analysis failed for one or more timeframes."
        )


if __name__ == "__main__":
    main()
