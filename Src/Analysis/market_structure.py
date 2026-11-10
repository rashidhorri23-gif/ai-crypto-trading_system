
import numpy as np
import pandas as pd

from indicators import validate_candles


DEFAULT_LEFT_BARS = 3
DEFAULT_RIGHT_BARS = 3


def detect_swings(
    data,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS
):
    if not isinstance(left_bars, int) or isinstance(left_bars, bool):
        raise ValueError("left_bars must be an integer.")

    if not isinstance(right_bars, int) or isinstance(right_bars, bool):
        raise ValueError("right_bars must be an integer.")

    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing periods must be positive.")

    result = validate_candles(data)

    count = len(result)

    result["swing_high_confirmed"] = False
    result["swing_low_confirmed"] = False

    result["swing_high_price"] = np.nan
    result["swing_low_price"] = np.nan

    result["swing_high_pivot_timestamp"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result["swing_low_pivot_timestamp"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result["swing_high_confirmation_timestamp"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result["swing_low_confirmation_timestamp"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    high_values = result["high"].to_numpy(dtype=float)
    low_values = result["low"].to_numpy(dtype=float)
    timestamps = result["timestamp"].to_numpy(dtype=np.int64)

    first_confirmation_index = left_bars + right_bars

    for current_index in range(first_confirmation_index, count):
        pivot_index = current_index - right_bars

        window_start = pivot_index - left_bars
        window_end = current_index + 1

        window_highs = high_values[window_start:window_end]
        window_lows = low_values[window_start:window_end]

        pivot_high = high_values[pivot_index]
        pivot_low = low_values[pivot_index]

        highest_value = np.max(window_highs)
        lowest_value = np.min(window_lows)

        unique_high = np.count_nonzero(
            window_highs == highest_value
        ) == 1

        unique_low = np.count_nonzero(
            window_lows == lowest_value
        ) == 1

        if pivot_high == highest_value and unique_high:
            result.at[current_index, "swing_high_confirmed"] = True
            result.at[current_index, "swing_high_price"] = pivot_high
            result.at[current_index, "swing_high_pivot_timestamp"] = int(
                timestamps[pivot_index]
            )
            result.at[
                current_index,
                "swing_high_confirmation_timestamp"
            ] = int(timestamps[current_index])

        if pivot_low == lowest_value and unique_low:
            result.at[current_index, "swing_low_confirmed"] = True
            result.at[current_index, "swing_low_price"] = pivot_low
            result.at[current_index, "swing_low_pivot_timestamp"] = int(
                timestamps[pivot_index]
            )
            result.at[
                current_index,
                "swing_low_confirmation_timestamp"
            ] = int(timestamps[current_index])

    return result


def detect_market_structure(
    data,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS
):
    result = detect_swings(
        data,
        left_bars=left_bars,
        right_bars=right_bars
    )

    count = len(result)

    result["structure_event"] = "NONE"
    result["structure_direction"] = 0
    result["structure_trend"] = 0

    result["broken_level"] = np.nan
    result["broken_level_timestamp"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result["structure_event_timestamp"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    latest_swing_high = None
    latest_swing_low = None

    latest_high_timestamp = None
    latest_low_timestamp = None

    high_already_broken = False
    low_already_broken = False

    current_trend = 0

    for index in range(count):
        candle = result.iloc[index]
        close_price = float(candle["close"])
        current_timestamp = int(candle["timestamp"])

        if bool(candle["swing_high_confirmed"]):
            latest_swing_high = float(
                candle["swing_high_price"]
            )

            latest_high_timestamp = int(
                candle["swing_high_pivot_timestamp"]
            )

            high_already_broken = False

        if bool(candle["swing_low_confirmed"]):
            latest_swing_low = float(
                candle["swing_low_price"]
            )

            latest_low_timestamp = int(
                candle["swing_low_pivot_timestamp"]
            )

            low_already_broken = False

        bullish_break = (
            latest_swing_high is not None
            and not high_already_broken
            and close_price > latest_swing_high
        )

        bearish_break = (
            latest_swing_low is not None
            and not low_already_broken
            and close_price < latest_swing_low
        )

        if bullish_break and bearish_break:
            raise RuntimeError(
                "Conflicting bullish and bearish structure breaks."
            )

        event = "NONE"
        direction = 0
        broken_level = np.nan
        broken_timestamp = None

        if bullish_break:
            high_already_broken = True

            direction = 1
            broken_level = latest_swing_high
            broken_timestamp = latest_high_timestamp

            if current_trend == -1:
                event = "CHOCH_BULLISH"
            else:
                event = "BOS_BULLISH"

            current_trend = 1

        elif bearish_break:
            low_already_broken = True

            direction = -1
            broken_level = latest_swing_low
            broken_timestamp = latest_low_timestamp

            if current_trend == 1:
                event = "CHOCH_BEARISH"
            else:
                event = "BOS_BEARISH"

            current_trend = -1

        result.at[index, "structure_event"] = event
        result.at[index, "structure_direction"] = direction
        result.at[index, "structure_trend"] = current_trend

        if event != "NONE":
            result.at[index, "broken_level"] = broken_level

            result.at[
                index,
                "broken_level_timestamp"
            ] = broken_timestamp

            result.at[
                index,
                "structure_event_timestamp"
            ] = current_timestamp

    return result
