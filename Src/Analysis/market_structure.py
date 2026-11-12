
import numpy as np
import pandas as pd

try:
    from .daily_structure import (
        analyze_daily_structure,
        detect_swings as detect_confirmed_swings,
    )
except ImportError:
    from daily_structure import (
        analyze_daily_structure,
        detect_swings as detect_confirmed_swings,
    )


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
        column for column in REQUIRED_COLUMNS
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

    for column in ["open", "high", "low", "close"]:
        if (result[column] <= 0).any():
            raise ValueError(
                f"{column.capitalize()} prices must be positive."
            )

    invalid_ohlc = (
        (result["high"] < result["low"])
        | (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            "Invalid OHLC candle relationships found."
        )

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

    median_timestamp = int(np.median(timestamps))
    median_duration = int(np.median(differences))

    if median_timestamp >= 100_000_000_000:
        if median_duration % 1000 != 0:
            raise ValueError(
                "Invalid millisecond candle interval."
            )

        median_duration //= 1000

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


def scale_analysis_timestamps(value, factor):
    if isinstance(value, dict):
        result = {}

        for key, item in value.items():
            if "timestamp" in key.lower():
                if (
                    isinstance(item, (int, float, np.integer, np.floating))
                    and not isinstance(item, bool)
                ):
                    result[key] = int(item * factor)
                else:
                    result[key] = item
            else:
                result[key] = scale_analysis_timestamps(
                    item,
                    factor
                )

        return result

    if isinstance(value, list):
        return [
            scale_analysis_timestamps(item, factor)
            for item in value
        ]

    return value


def get_event_direction(event_name):
    if not isinstance(event_name, str):
        return 0

    if event_name.endswith("BULLISH"):
        return 1

    if event_name.endswith("BEARISH"):
        return -1

    return 0


def create_analysis(
    data,
    timeframe=None,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS,
):
    for name, value in (
        ("left_bars", left_bars),
        ("right_bars", right_bars),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    validated = validate_input_data(data)

    timestamp_is_milliseconds = (
        int(validated["timestamp"].median())
        >= 100_000_000_000
    )

    if timeframe is None:
        timeframe = infer_timeframe(validated)
    else:
        timeframe = validate_timeframe(timeframe)

    engine_data = validated.copy()

    if timestamp_is_milliseconds:
        if (engine_data["timestamp"] % 1000 != 0).any():
            raise ValueError(
                "Millisecond timestamps must align with whole seconds."
            )

        engine_data["timestamp"] = (
            engine_data["timestamp"] // 1000
        )

    candles = prepare_candles(engine_data)

    analysis = analyze_daily_structure(
        candles,
        timeframe=timeframe,
        internal_left_bars=left_bars,
        internal_right_bars=right_bars,
    )

    if not isinstance(analysis, dict):
        raise TypeError(
            "Daily structure analysis must return a dictionary."
        )

    if analysis.get("status") != "ANALYZED":
        raise RuntimeError(
            "Daily structure analysis did not complete successfully."
        )

    if timestamp_is_milliseconds:
        analysis = scale_analysis_timestamps(
            analysis,
            1000
        )

    return validated, analysis


def _initialize_swing_columns(result):
    count = len(result)

    columns = [
        "internal_swing_high_confirmed",
        "internal_swing_low_confirmed",
        "external_swing_high_confirmed",
        "external_swing_low_confirmed",
        "swing_high_confirmed",
        "swing_low_confirmed",
    ]

    for column in columns:
        result[column] = False

    price_columns = [
        "internal_swing_high_price",
        "internal_swing_low_price",
        "external_swing_high_price",
        "external_swing_low_price",
        "swing_high_price",
        "swing_low_price",
    ]

    for column in price_columns:
        result[column] = np.nan

    timestamp_columns = [
        "internal_swing_high_pivot_timestamp",
        "internal_swing_low_pivot_timestamp",
        "external_swing_high_pivot_timestamp",
        "external_swing_low_pivot_timestamp",
        "swing_high_pivot_timestamp",
        "swing_low_pivot_timestamp",
        "swing_high_confirmation_timestamp",
        "swing_low_confirmation_timestamp",
    ]

    for column in timestamp_columns:
        result[column] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64"
        )

    return result


def add_swing_columns(result, analysis):
    result = _initialize_swing_columns(result)
    count = len(result)

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

            if swing_type not in ("HIGH", "LOW"):
                raise ValueError(
                    f"Unknown swing type: {swing_type}"
                )

            confirmed_column = (
                f"{structure_name}_swing_{swing_type.lower()}_confirmed"
            )

            price_column = (
                f"{structure_name}_swing_{swing_type.lower()}_price"
            )

            timestamp_column = (
                f"{structure_name}_swing_{swing_type.lower()}_pivot_timestamp"
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

            if swing_type == "HIGH":
                result.at[
                    confirmation_index,
                    "swing_high_confirmed"
                ] = True

                result.at[
                    confirmation_index,
                    "swing_high_price"
                ] = price

                result.at[
                    confirmation_index,
                    "swing_high_pivot_timestamp"
                ] = pivot_timestamp

                result.at[
                    confirmation_index,
                    "swing_high_confirmation_timestamp"
                ] = int(result.at[confirmation_index, "timestamp"])

            else:
                result.at[
                    confirmation_index,
                    "swing_low_confirmed"
                ] = True

                result.at[
                    confirmation_index,
                    "swing_low_price"
                ] = price

                result.at[
                    confirmation_index,
                    "swing_low_pivot_timestamp"
                ] = pivot_timestamp

                result.at[
                    confirmation_index,
                    "swing_low_confirmation_timestamp"
                ] = int(result.at[confirmation_index, "timestamp"])

    return result


def add_direct_swing_columns(result, swings):
    result = _initialize_swing_columns(result)
    count = len(result)

    for swing in swings:
        confirmation_index = int(
            swing["confirmation_index"]
        )

        if not 0 <= confirmation_index < count:
            raise ValueError(
                "Swing confirmation index is outside candle data."
            )

        swing_type = str(swing["type"]).upper()
        price = float(swing["price"])
        pivot_timestamp = int(swing["timestamp"])

        if swing_type not in ("HIGH", "LOW"):
            raise ValueError(
                f"Unknown swing type: {swing_type}"
            )

        prefix = f"internal_swing_{swing_type.lower()}"

        result.at[
            confirmation_index,
            f"{prefix}_confirmed"
        ] = True

        result.at[
            confirmation_index,
            f"{prefix}_price"
        ] = price

        result.at[
            confirmation_index,
            f"{prefix}_pivot_timestamp"
        ] = pivot_timestamp

        result.at[
            confirmation_index,
            f"swing_{swing_type.lower()}_confirmed"
        ] = True

        result.at[
            confirmation_index,
            f"swing_{swing_type.lower()}_price"
        ] = price

        result.at[
            confirmation_index,
            f"swing_{swing_type.lower()}_pivot_timestamp"
        ] = pivot_timestamp

        result.at[
            confirmation_index,
            f"swing_{swing_type.lower()}_confirmation_timestamp"
        ] = int(result.at[confirmation_index, "timestamp"])

    return result


def _initialize_event_columns(result):
    event_names = [
        "internal_structure_event",
        "external_structure_event",
        "structure_event",
    ]

    for column in event_names:
        result[column] = "NONE"

    direction_columns = [
        "internal_structure_direction",
        "external_structure_direction",
        "structure_direction",
        "internal_structure_trend",
        "external_structure_trend",
        "structure_trend",
    ]

    for column in direction_columns:
        result[column] = 0

    price_columns = [
        "internal_broken_level",
        "external_broken_level",
        "broken_level",
    ]

    for column in price_columns:
        result[column] = np.nan

    timestamp_columns = [
        "internal_broken_level_timestamp",
        "external_broken_level_timestamp",
        "internal_event_available_timestamp",
        "external_event_available_timestamp",
        "broken_level_timestamp",
        "structure_event_timestamp",
    ]

    for column in timestamp_columns:
        result[column] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64"
        )

    return result


def _safe_timestamp(value):
    if value is None or value is pd.NA:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        return int(value)

    return None


def _store_structure_events(result, structure_name, structure):
    count = len(result)
    event_column = f"{structure_name}_structure_event"
    direction_column = f"{structure_name}_structure_direction"
    level_column = f"{structure_name}_broken_level"
    level_timestamp_column = (
        f"{structure_name}_broken_level_timestamp"
    )
    available_column = (
        f"{structure_name}_event_available_timestamp"
    )

    for event in structure.get("events", []):
        event_index = int(event["index"])

        if not 0 <= event_index < count:
            raise ValueError(
                "Structure event index is outside candle data."
            )

        event_name = event.get("event", "NONE")
        direction = get_event_direction(event_name)

        result.at[event_index, event_column] = event_name
        result.at[event_index, direction_column] = direction

        level = event.get("level")

        if level is not None and _safe_timestamp(level) is None:
            result.at[event_index, level_column] = float(level)

        level_timestamp = _safe_timestamp(
            event.get("level_timestamp")
        )

        if level_timestamp is None:
            level_timestamp = _safe_timestamp(
                event.get("broken_swing_timestamp")
            )

        if level_timestamp is not None:
            result.at[
                event_index,
                level_timestamp_column
            ] = level_timestamp

        available_timestamp = _safe_timestamp(
            event.get("available_timestamp")
        )

        if available_timestamp is not None:
            result.at[
                event_index,
                available_column
            ] = available_timestamp


def add_event_columns(result, analysis):
    result = _initialize_event_columns(result)

    structures = {
        "internal": analysis["internal_structure"],
        "external": analysis["external_structure"],
    }

    for structure_name, structure in structures.items():
        _store_structure_events(
            result,
            structure_name,
            structure
        )

    internal_trend = 0
    external_trend = 0

    for index in result.index:
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

        if internal_event != "NONE":
            internal_trend = internal_direction

        if external_event != "NONE":
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
            result.at[index, "structure_event"] = external_event

        elif internal_event != "NONE":
            result.at[index, "structure_event"] = internal_event

        if external_event != "NONE":
            selected = "external"
            selected_direction = external_direction
        elif internal_event != "NONE":
            selected = "internal"
            selected_direction = internal_direction
        else:
            selected = None
            selected_direction = 0

        if selected is not None:
            result.at[
                index,
                "structure_direction"
            ] = selected_direction

            result.at[
                index,
                "broken_level"
            ] = result.at[
                index,
                f"{selected}_broken_level"
            ]

            level_timestamp = _safe_timestamp(
                result.at[
                    index,
                    f"{selected}_broken_level_timestamp"
                ]
            )

            if level_timestamp is not None:
                result.at[
                    index,
                    "broken_level_timestamp"
                ] = level_timestamp

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
    if (
        isinstance(left_bars, bool)
        or not isinstance(left_bars, int)
        or left_bars < 1
    ):
        raise ValueError("left_bars must be a positive integer.")

    if (
        isinstance(right_bars, bool)
        or not isinstance(right_bars, int)
        or right_bars < 1
    ):
        raise ValueError("right_bars must be a positive integer.")

    validated = validate_input_data(data)
    candles = prepare_candles(validated)

    swings = detect_confirmed_swings(
        candles,
        left_bars=left_bars,
        right_bars=right_bars
    )

    return add_direct_swing_columns(
        validated.copy(),
        swings
    )


def detect_market_structure(
    data,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS,
    timeframe=None
):
    if (
        isinstance(left_bars, bool)
        or not isinstance(left_bars, int)
        or left_bars < 1
    ):
        raise ValueError("left_bars must be a positive integer.")

    if (
        isinstance(right_bars, bool)
        or not isinstance(right_bars, int)
        or right_bars < 1
    ):
        raise ValueError("right_bars must be a positive integer.")

    result, analysis = create_analysis(
        data,
        timeframe=timeframe,
        left_bars=left_bars,
        right_bars=right_bars,
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
