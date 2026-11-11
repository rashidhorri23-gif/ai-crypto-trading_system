
import numpy as np
import pandas as pd

try:
    from .daily_structure import analyze_daily_structure
except ImportError:
    from daily_structure import analyze_daily_structure


DEFAULT_LEFT_BARS = 3
DEFAULT_RIGHT_BARS = 3

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

SUPPORTED_TIMEFRAMES = {
    300: "5m",
    900: "15m",
    3600: "1H",
    14400: "4H",
    86400: "1D",
}


def validate_input_data(data):
    if isinstance(data, pd.DataFrame):
        result = data.copy()
    elif isinstance(data, list):
        result = pd.DataFrame(data)
    else:
        raise TypeError(
            "Data must be a pandas DataFrame or a list of candle dictionaries."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required candle columns: {missing}"
        )

    result = result[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise"
        )

    values = result[REQUIRED_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Candle data contains NaN or infinite values."
        )

    if (result["timestamp"] < 0).any():
        raise ValueError("Timestamps cannot be negative.")

    if (result["open"] <= 0).any():
        raise ValueError("Open prices must be positive.")

    if (result["high"] <= 0).any():
        raise ValueError("High prices must be positive.")

    if (result["low"] <= 0).any():
        raise ValueError("Low prices must be positive.")

    if (result["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    invalid_ohlc = (
        (result["high"] < result["low"])
        | (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC candle relationships found.")

    result["timestamp"] = result["timestamp"].astype("int64")

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps found.")

    if len(result) < 20:
        raise ValueError(
            "At least 20 candles are required for structure analysis."
        )

    return result


def infer_timeframe(data):
    timestamps = data["timestamp"].to_numpy(dtype=np.int64)

    differences = np.diff(timestamps)

    if len(differences) == 0:
        raise ValueError(
            "Not enough timestamps to infer timeframe."
        )

    if np.any(differences <= 0):
        raise ValueError(
            "Timestamps must be strictly increasing."
        )

    median_duration = int(np.median(differences))

    if median_duration not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Unsupported inferred candle duration: {median_duration} seconds."
        )

    return SUPPORTED_TIMEFRAMES[median_duration]


def validate_timeframe(timeframe):
    if timeframe not in SUPPORTED_TIMEFRAMES.values():
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    return timeframe


def prepare_candles(data):
    return [
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


def get_event_direction(event_name):
    if not isinstance(event_name, str):
        return 0

    if event_name.endswith("BULLISH"):
        return 1

    if event_name.endswith("BEARISH"):
        return -1

    return 0


def create_analysis(data, timeframe=None):
    validated = validate_input_data(data)

    if timeframe is None:
        timeframe = infer_timeframe(validated)
    else:
        timeframe = validate_timeframe(timeframe)

    candles = prepare_candles(validated)

    analysis = analyze_daily_structure(
        candles,
        timeframe=timeframe
    )

    if analysis.get("status") != "ANALYZED":
        raise RuntimeError(
            "Daily structure analysis did not complete successfully."
        )

    return validated, analysis


def add_swing_columns(result, analysis):
    count = len(result)

    result["internal_swing_high_confirmed"] = False
    result["internal_swing_low_confirmed"] = False

    result["external_swing_high_confirmed"] = False
    result["external_swing_low_confirmed"] = False

    result["internal_swing_high_price"] = np.nan
    result["internal_swing_low_price"] = np.nan

    result["external_swing_high_price"] = np.nan
    result["external_swing_low_price"] = np.nan

    result["internal_swing_high_pivot_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["internal_swing_low_pivot_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["external_swing_high_pivot_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["external_swing_low_pivot_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    structures = {
        "internal": analysis["internal_structure"],
        "external": analysis["external_structure"],
    }

    for structure_name, structure in structures.items():
        for swing in structure.get("swings", []):
            confirmation_index = int(
                swing["confirmation_index"]
            )

            if not 0 <= confirmation_index < count:
                raise ValueError(
                    "Swing confirmation index is outside candle data."
                )

            swing_type = swing["type"]
            price = float(swing["price"])
            pivot_timestamp = int(swing["timestamp"])

            if swing_type == "HIGH":
                confirmed_column = (
                    f"{structure_name}_swing_high_confirmed"
                )

                price_column = (
                    f"{structure_name}_swing_high_price"
                )

                timestamp_column = (
                    f"{structure_name}_swing_high_pivot_timestamp"
                )

            elif swing_type == "LOW":
                confirmed_column = (
                    f"{structure_name}_swing_low_confirmed"
                )

                price_column = (
                    f"{structure_name}_swing_low_price"
                )

                timestamp_column = (
                    f"{structure_name}_swing_low_pivot_timestamp"
                )

            else:
                raise ValueError(
                    f"Unknown swing type: {swing_type}"
                )

            result.at[
                confirmation_index,
                confirmed_column
            ] = True

            result.at[
                confirmation_index,
                price_column
            ] = price

            result.at[
                confirmation_index,
                timestamp_column
            ] = pivot_timestamp

    result["swing_high_confirmed"] = (
        result["internal_swing_high_confirmed"]
        | result["external_swing_high_confirmed"]
    )

    result["swing_low_confirmed"] = (
        result["internal_swing_low_confirmed"]
        | result["external_swing_low_confirmed"]
    )

    result["swing_high_price"] = (
        result["external_swing_high_price"].combine_first(
            result["internal_swing_high_price"]
        )
    )

    result["swing_low_price"] = (
        result["external_swing_low_price"].combine_first(
            result["internal_swing_low_price"]
        )
    )

    result["swing_high_pivot_timestamp"] = (
        result["external_swing_high_pivot_timestamp"].combine_first(
            result["internal_swing_high_pivot_timestamp"]
        )
    )

    result["swing_low_pivot_timestamp"] = (
        result["external_swing_low_pivot_timestamp"].combine_first(
            result["internal_swing_low_pivot_timestamp"]
        )
    )

    result["swing_high_confirmation_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["swing_low_confirmation_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    for index in result.index:
        if result.at[index, "swing_high_confirmed"]:
            result.at[
                index,
                "swing_high_confirmation_timestamp"
            ] = int(result.at[index, "timestamp"])

        if result.at[index, "swing_low_confirmed"]:
            result.at[
                index,
                "swing_low_confirmation_timestamp"
            ] = int(result.at[index, "timestamp"])

    return result


def add_event_columns(result, analysis):
    count = len(result)

    event_columns = [
        "internal_structure_event",
        "external_structure_event",
    ]

    for column in event_columns:
        result[column] = "NONE"

    result["internal_structure_direction"] = 0
    result["external_structure_direction"] = 0

    result["internal_broken_level"] = np.nan
    result["external_broken_level"] = np.nan

    result["internal_broken_level_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["external_broken_level_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["internal_event_available_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["external_event_available_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    internal_events = {}
    external_events = {}

    structures = {
        "internal": analysis["internal_structure"],
        "external": analysis["external_structure"],
    }

    for structure_name, structure in structures.items():
        for event in structure.get("events", []):
            event_index = int(event["index"])

            if not 0 <= event_index < count:
                raise ValueError(
                    "Structure event index is outside candle data."
                )

            if structure_name == "internal":
                internal_events.setdefault(
                    event_index,
                    []
                ).append(event)
            else:
                external_events.setdefault(
                    event_index,
                    []
                ).append(event)

    for index, events in internal_events.items():
        event_names = [
            event["event"]
            for event in events
        ]

        result.at[
            index,
            "internal_structure_event"
        ] = "|".join(event_names)

        result.at[
            index,
            "internal_structure_direction"
        ] = get_event_direction(event_names[-1])

        selected_event = events[-1]

        result.at[
            index,
            "internal_broken_level"
        ] = selected_event.get("level", np.nan)

        level_timestamp = selected_event.get(
            "level_timestamp"
        )

        if level_timestamp is not None:
            result.at[
                index,
                "internal_broken_level_timestamp"
            ] = int(level_timestamp)

        available_timestamp = selected_event.get(
            "available_timestamp"
        )

        if available_timestamp is not None:
            result.at[
                index,
                "internal_event_available_timestamp"
            ] = int(available_timestamp)

    for index, events in external_events.items():
        event_names = [
            event["event"]
            for event in events
        ]

        result.at[
            index,
            "external_structure_event"
        ] = "|".join(event_names)

        result.at[
            index,
            "external_structure_direction"
        ] = get_event_direction(event_names[-1])

        selected_event = events[-1]

        result.at[
            index,
            "external_broken_level"
        ] = selected_event.get("level", np.nan)

        level_timestamp = selected_event.get(
            "level_timestamp"
        )

        if level_timestamp is not None:
            result.at[
                index,
                "external_broken_level_timestamp"
            ] = int(level_timestamp)

        available_timestamp = selected_event.get(
            "available_timestamp"
        )

        if available_timestamp is not None:
            result.at[
                index,
                "external_event_available_timestamp"
            ] = int(available_timestamp)

    result["structure_event"] = "NONE"
    result["structure_direction"] = 0
    result["structure_trend"] = 0

    result["broken_level"] = np.nan

    result["broken_level_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["structure_event_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    internal_trend = 0
    external_trend = 0

    result["internal_structure_trend"] = 0
    result["external_structure_trend"] = 0

    for index in range(count):
        internal_event = result.at[
            index,
            "internal_structure_event"
        ]

        external_event = result.at[
            index,
            "external_structure_event"
        ]

        internal_direction = int(
            result.at[
                index,
                "internal_structure_direction"
            ]
        )

        external_direction = int(
            result.at[
                index,
                "external_structure_direction"
            ]
        )

        if internal_event != "NONE" and internal_direction != 0:
            internal_trend = internal_direction

        if external_event != "NONE" and external_direction != 0:
            external_trend = external_direction

        result.at[
            index,
            "internal_structure_trend"
        ] = internal_trend

        result.at[
            index,
            "external_structure_trend"
        ] = external_trend

        if internal_event != "NONE" and external_event != "NONE":
            result.at[index, "structure_event"] = (
                f"INTERNAL:{internal_event}|EXTERNAL:{external_event}"
            )

        elif external_event != "NONE":
            result.at[
                index,
                "structure_event"
            ] = external_event

        elif internal_event != "NONE":
            result.at[
                index,
                "structure_event"
            ] = internal_event

        if external_event != "NONE":
            selected_event = "external"
            selected_direction = external_direction

        elif internal_event != "NONE":
            selected_event = "internal"
            selected_direction = internal_direction

        else:
            selected_event = None
            selected_direction = 0

        if selected_event is not None:
            result.at[
                index,
                "structure_direction"
            ] = selected_direction

            result.at[
                index,
                "broken_level"
            ] = result.at[
                index,
                f"{selected_event}_broken_level"
            ]

            result.at[
                index,
                "broken_level_timestamp"
            ] = result.at[
                index,
                f"{selected_event}_broken_level_timestamp"
            ]

            result.at[
                index,
                "structure_event_timestamp"
            ] = int(result.at[index, "timestamp"])

        result.at[
            index,
            "structure_trend"
        ] = external_trend

    return result


def detect_swings(
    data,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS,
    timeframe=None
):
    if not isinstance(left_bars, int) or isinstance(left_bars, bool):
        raise ValueError("left_bars must be an integer.")

    if not isinstance(right_bars, int) or isinstance(right_bars, bool):
        raise ValueError("right_bars must be an integer.")

    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing periods must be positive.")

    result, analysis = create_analysis(
        data,
        timeframe=timeframe
    )

    result = add_swing_columns(
        result,
        analysis
    )

    return result


def detect_market_structure(
    data,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS,
    timeframe=None
):
    if not isinstance(left_bars, int) or isinstance(left_bars, bool):
        raise ValueError("left_bars must be an integer.")

    if not isinstance(right_bars, int) or isinstance(right_bars, bool):
        raise ValueError("right_bars must be an integer.")

    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing periods must be positive.")

    result, analysis = create_analysis(
        data,
        timeframe=timeframe
    )

    result = add_swing_columns(
        result,
        analysis
    )

    result = add_event_columns(
        result,
        analysis
    )

    return result
