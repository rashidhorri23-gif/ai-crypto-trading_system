
from pathlib import Path
import sys

import numpy as np
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

TIMEFRAME_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1H": 3600,
    "4H": 14400,
    "1D": 86400,
}

LEFT_BARS = 3
RIGHT_BARS = 3


def load_archive(timeframe):

    file_path = (
        ARCHIVE_DIR
        / f"BTC_USDT_SPOT_{timeframe}.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {file_path}"
        )

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
        column
        for column in required
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

    values = data[required].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"NaN or infinite values found in {timeframe}"
        )

    if (data["timestamp"] < 0).any():
        raise ValueError(
            f"Negative timestamps found in {timeframe}"
        )

    if (data["open"] <= 0).any():
        raise ValueError(
            f"Invalid open prices in {timeframe}"
        )

    if (data["high"] <= 0).any():
        raise ValueError(
            f"Invalid high prices in {timeframe}"
        )

    if (data["low"] <= 0).any():
        raise ValueError(
            f"Invalid low prices in {timeframe}"
        )

    if (data["close"] <= 0).any():
        raise ValueError(
            f"Invalid close prices in {timeframe}"
        )

    if (data["volume"] < 0).any():
        raise ValueError(
            f"Negative volume found in {timeframe}"
        )

    invalid_ohlc = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            f"Invalid OHLC relationships in {timeframe}"
        )

    data["timestamp"] = data["timestamp"].astype("int64")

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if data["timestamp"].duplicated().any():
        raise ValueError(
            f"Duplicate timestamps found in {timeframe}"
        )

    if len(data) < 20:
        raise ValueError(
            f"Insufficient candles in {timeframe}"
        )

    return data


def collect_structure_events(analysis, timeframe):

    collected = []

    structures = {
        "INTERNAL": analysis["internal_structure"],
        "EXTERNAL": analysis["external_structure"],
    }

    candle_duration = TIMEFRAME_SECONDS[timeframe]

    for structure_name, structure in structures.items():

        events = structure.get("events", [])

        for event in events:

            event_type = event.get("event")

            if event_type not in EVENT_TYPES:
                continue

            try:
                event_index = int(event["index"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"Invalid event index in {structure_name}"
                )

            if event_index < 0:
                raise ValueError(
                    "Negative event index detected."
                )

            event_timestamp = event.get("timestamp")

            available_timestamp = event.get(
                "available_timestamp"
            )

            collected.append({
                "structure": structure_name,
                "event": event_type,
                "index": event_index,
                "timestamp": (
                    int(event_timestamp)
                    if event_timestamp is not None
                    else None
                ),
                "available_timestamp": (
                    int(available_timestamp)
                    if available_timestamp is not None
                    else None
                ),
                "level": event.get("level"),
                "candle_duration": candle_duration,
            })

    return collected


def calculate_outcome(
    data,
    signal_index,
    direction,
    horizon,
    available_timestamp=None
):

    entry_index = signal_index + 1

    if entry_index >= len(data):
        return None

    end_index = entry_index + horizon

    if end_index > len(data):
        return None

    entry_candle = data.iloc[entry_index]

    entry_timestamp = int(
        entry_candle["timestamp"]
    )

    if (
        available_timestamp is not None
        and entry_timestamp < available_timestamp
    ):
        raise ValueError(
            "Entry candle starts before signal availability."
        )

    entry_price = float(
        entry_candle["open"]
    )

    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(
            "Invalid entry price."
        )

    future = data.iloc[
        entry_index:end_index
    ]

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
            (final_close - entry_price)
            / entry_price
        ) * 100

        mfe = (
            (highest_price - entry_price)
            / entry_price
        ) * 100

        mae = (
            (entry_price - lowest_price)
            / entry_price
        ) * 100

    elif direction == -1:

        directional_return = (
            (entry_price - final_close)
            / entry_price
        ) * 100

        mfe = (
            (entry_price - lowest_price)
            / entry_price
        ) * 100

        mae = (
            (highest_price - entry_price)
            / entry_price
        ) * 100

    else:
        raise ValueError(
            "Direction must be 1 or -1."
        )

    return {
        "entry_price": entry_price,
        "entry_timestamp": entry_timestamp,
        "final_close": final_close,
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

        result = calculate_outcome(
            data=data,
            signal_index=event["index"],
            direction=direction,
            horizon=horizon,
            available_timestamp=event[
                "available_timestamp"
            ],
        )

        if result is not None:
            outcomes.append(result)

    if not outcomes:
        return None

    results = pd.DataFrame(outcomes)

    return {
        "observations": len(results),
        "wins": int(results["win"].sum()),
        "losses": int((~results["win"]).sum()),
        "win_rate": (
            results["win"].mean() * 100
        ),
        "average_return": (
            results["return"].mean()
        ),
        "median_return": (
            results["return"].median()
        ),
        "average_mfe": (
            results["mfe"].mean()
        ),
        "average_mae": (
            results["mae"].mean()
        ),
    }


def analyze_timeframe(timeframe):

    data = load_archive(timeframe)

    candles = [
        {
            "timestamp": int(row.timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in data.itertuples(index=False)
    ]

    analysis = analyze_daily_structure(
        candles,
        timeframe=timeframe
    )

    if not isinstance(analysis, dict):
        raise ValueError(
            "Invalid daily structure analysis result."
        )

    if (
        "internal_structure" not in analysis
        or "external_structure" not in analysis
    ):
        raise ValueError(
            "Internal or external structure missing."
        )

    events = collect_structure_events(
        analysis,
        timeframe
    )

    for event in events:

        if event["index"] >= len(data):
            raise ValueError(
                "Event index exceeds archive length."
            )

        actual_timestamp = int(
            data.iloc[event["index"]]["timestamp"]
        )

        if (
            event["timestamp"] is not None
            and event["timestamp"] != actual_timestamp
        ):
            raise ValueError(
                "Event timestamp does not match archive index."
            )

        if event["available_timestamp"] is None:
            event["available_timestamp"] = (
                actual_timestamp
                + event["candle_duration"]
            )

    print()
    print("=" * 75)
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(data)}")
    print(f"Total Structure Events: {len(events)}")
    print("=" * 75)

    structure_names = [
        "INTERNAL",
        "EXTERNAL",
    ]

    total_evaluations = 0
    total_eligible_events = 0

    for structure_name in structure_names:

        structure_events = [
            event
            for event in events
            if event["structure"] == structure_name
        ]

        print()
        print("-" * 75)
        print(f"STRUCTURE: {structure_name}")
        print(f"Total Events: {len(structure_events)}")
        print("-" * 75)

        for event_type in EVENT_TYPES:

            event_group = [
                event
                for event in structure_events
                if event["event"] == event_type
            ]

            print()
            print(f"EVENT: {event_type}")
            print(f"Signal Count: {len(event_group)}")

            for direction_name, direction_value in DIRECTIONS.items():

                print()
                print(f"DIRECTION: {direction_name}")

                for horizon in HORIZONS:

                    result = evaluate_event_type(
                        data=data,
                        events=event_group,
                        direction=direction_value,
                        horizon=horizon,
                    )

                    if result is None:
                        print(
                            f"Horizon {horizon}: "
                            "No eligible observations"
                        )
                        continue

                    total_evaluations += 1

                    if horizon == HORIZONS[0]:
                        total_eligible_events += (
                            result["observations"]
                        )

                    print(
                        f"Horizon {horizon}: "
                        f"N={result['observations']}, "
                        f"Wins={result['wins']}, "
                        f"Losses={result['losses']}, "
                        f"WinRate={result['win_rate']:.2f}%, "
                        f"AvgReturn={result['average_return']:.5f}%, "
                        f"MedianReturn={result['median_return']:.5f}%, "
                        f"AvgMFE={result['average_mfe']:.5f}%, "
                        f"AvgMAE={result['average_mae']:.5f}%"
                    )

    print()
    print("=" * 75)
    print(f"TIMEFRAME SUMMARY: {timeframe}")
    print(f"Internal Events: {sum(1 for e in events if e['structure'] == 'INTERNAL')}")
    print(f"External Events: {sum(1 for e in events if e['structure'] == 'EXTERNAL')}")
    print(f"Eligible Event Observations: {total_eligible_events}")
    print(f"Evaluation Groups: {total_evaluations}")
    print("=" * 75)

    return {
        "timeframe": timeframe,
        "candles": len(data),
        "events": len(events),
        "eligible_events": total_eligible_events,
        "evaluation_groups": total_evaluations,
    }


def main():

    print()
    print("=" * 75)
    print("BOS AND CHOCH LONG / SHORT EVENT STUDY")
    print("=" * 75)

    print("Market: BTC-USDT SPOT")
    print("Data Source: Local OKX Archive")
    print("Structure Engine: DAILY_STRUCTURE")
    print("Structures: INTERNAL + EXTERNAL")
    print("Event Types: BOS + CHOCH")
    print("Directions: LONG + SHORT")
    print("Entry: Next Candle Open")
    print(f"Forward Horizons: {HORIZONS}")
    print("Swing Configuration: Engine Defaults")
    print("Fees: NOT INCLUDED")
    print("Slippage: NOT INCLUDED")
    print("Stop Loss / Take Profit: NOT INCLUDED")
    print("Overlapping Event Windows: NOT REMOVED")
    print("Trading Execution: DISABLED")
    print("Archive Modification: NO")

    successful = []
    failed = []

    for timeframe in TIMEFRAMES:

        try:

            result = analyze_timeframe(
                timeframe
            )

            successful.append(result)

        except Exception as error:

            failed.append({
                "timeframe": timeframe,
                "error": str(error),
            })

            print()
            print("=" * 75)
            print(f"TIMEFRAME FAILED: {timeframe}")
            print(f"ERROR: {error}")
            print("=" * 75)

    print()
    print("=" * 75)
    print("FINAL EXECUTION REPORT")
    print("=" * 75)

    print(
        f"Successful Timeframes: {len(successful)}"
    )

    print(
        f"Failed Timeframes: {len(failed)}"
    )

    print(
        f"Total Timeframes: {len(TIMEFRAMES)}"
    )

    if failed:

        print()
        print("FAILED TIMEFRAMES")

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
