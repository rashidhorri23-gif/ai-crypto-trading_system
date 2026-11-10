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
        pd.NA,
        index=result.index,
        dtype="Int64"
    )

    result["swing_low_pivot_timestamp"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64"
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

    high_values = result["high"].to_numpy(dtype=float)
    low_values = result["low"].to_numpy(dtype=float)
    timestamps = result["timestamp"].to_numpy(dtype=np.int64)

    first_confirmation_index = left_bars + right_bars

    for current_index in range(
        first_confirmation_index,
        count
    ):
        pivot_index = current_index - right_bars

        window_start = pivot_index - left_bars
        window_end = current_index + 1

        window_highs = high_values[
            window_start:window_end
        ]

        window_lows = low_values[
            window_start:window_end
        ]

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
            result.at[
                current_index,
                "swing_high_confirmed"
            ] = True

            result.at[
                current_index,
                "swing_high_price"
            ] = pivot_high

            result.at[
                current_index,
                "swing_high_pivot_timestamp"
            ] = int(timestamps[pivot_index])

            result.at[
                current_index,
                "swing_high_confirmation_timestamp"
            ] = int(timestamps[current_index])

        if pivot_low == lowest_value and unique_low:
            result.at[
                current_index,
                "swing_low_confirmed"
            ] = True

            result.at[
                current_index,
                "swing_low_price"
            ] = pivot_low

            result.at[
                current_index,
                "swing_low_pivot_timestamp"
            ] = int(timestamps[pivot_index])

            result.at[
                current_index,
                "swing_low_confirmation_timestamp"
            ] = int(timestamps[current_index])

    return result
