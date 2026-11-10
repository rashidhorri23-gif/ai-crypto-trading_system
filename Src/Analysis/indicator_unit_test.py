import sys
import math
import numpy as np
import pandas as pd

from pathlib import Path

from indicators import calculate_all_indicators


TEST_CANDLES = 300
TOLERANCE = 1e-8


def manual_ewm(values, alpha, min_periods):
    values = np.asarray(values, dtype=float)

    output = np.full(len(values), np.nan)

    state = None
    valid_count = 0

    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue

        if state is None:
            state = value
        else:
            state = (
                (1 - alpha) * state
                + alpha * value
            )

        valid_count += 1

        if valid_count >= min_periods:
            output[index] = state

    return output


def manual_ema(values, period):
    return manual_ewm(
        values,
        2 / (period + 1),
        period
    )


def manual_rsi(values, period=14):
    values = np.asarray(values, dtype=float)

    delta = np.diff(values, prepend=np.nan)

    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)

    average_gain = manual_ewm(
        gains,
        1 / period,
        period
    )

    average_loss = manual_ewm(
        losses,
        1 / period,
        period
    )

    result = np.full(len(values), np.nan)

    valid = np.isfinite(average_gain) & np.isfinite(average_loss)

    normal = valid & (average_loss > 0)

    relative_strength = (
        average_gain[normal]
        / average_loss[normal]
    )

    result[normal] = (
        100 - (100 / (1 + relative_strength))
    )

    result[valid & (average_loss == 0) & (average_gain > 0)] = 100

    result[valid & (average_loss == 0) & (average_gain == 0)] = 50

    return result


def manual_macd(values):
    ema_12 = manual_ema(values, 12)
    ema_26 = manual_ema(values, 26)

    macd_line = ema_12 - ema_26

    signal_line = manual_ewm(
        macd_line,
        2 / 10,
        9
    )

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def manual_atr(data, period=14):
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)

    previous_close = np.roll(close, 1)
    previous_close[0] = np.nan

    true_range = np.maximum.reduce([
        high - low,
        np.abs(high - previous_close),
        np.abs(low - previous_close)
    ])

    return manual_ewm(
        true_range,
        1 / period,
        period
    )


def manual_adx(data, period=14):
    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)

    upward_move = np.diff(high, prepend=np.nan)
    downward_move = -np.diff(low, prepend=np.nan)

    positive_dm = np.where(
        (upward_move > downward_move)
        & (upward_move > 0),
        upward_move,
        0
    )

    negative_dm = np.where(
        (downward_move > upward_move)
        & (downward_move > 0),
        downward_move,
        0
    )

    atr = manual_atr(data, period)

    positive_smoothed = manual_ewm(
        positive_dm,
        1 / period,
        period
    )

    negative_smoothed = manual_ewm(
        negative_dm,
        1 / period,
        period
    )

    positive_di = 100 * positive_smoothed / atr
    negative_di = 100 * negative_smoothed / atr

    denominator = positive_di + negative_di

    dx = np.full(len(data), np.nan)

    valid = (
        np.isfinite(denominator)
        & (denominator > 0)
    )

    dx[valid] = (
        100
        * np.abs(
            positive_di[valid] - negative_di[valid]
        )
        / denominator[valid]
    )

    return manual_ewm(
        dx,
        1 / period,
        period
    )


def manual_volume_analysis(data, period=20):
    volume = data["volume"].to_numpy(dtype=float)

    average = np.full(len(volume), np.nan)

    for index in range(period - 1, len(volume)):
        average[index] = np.mean(
            volume[index - period + 1:index + 1]
        )

    ratio = volume / average

    return average, ratio


def manual_vwap(data):
    timestamps = pd.to_datetime(
        data["timestamp"],
        unit="ms",
        utc=True
    )

    days = timestamps.dt.floor("D")

    typical_price = (
        data["high"].to_numpy(dtype=float)
        + data["low"].to_numpy(dtype=float)
        + data["close"].to_numpy(dtype=float)
    ) / 3

    volume = data["volume"].to_numpy(dtype=float)

    result = np.full(len(data), np.nan)

    daily_value = {}
    daily_volume = {}

    for index, day in enumerate(days):
        key = day

        daily_value[key] = (
            daily_value.get(key, 0)
            + typical_price[index] * volume[index]
        )

        daily_volume[key] = (
            daily_volume.get(key, 0)
            + volume[index]
        )

        if daily_volume[key] > 0:
            result[index] = (
                daily_value[key] / daily_volume[key]
            )

    return result


def build_test_data():
    index = np.arange(TEST_CANDLES)

    close = (
        50000
        + index * 3
        + 80 * np.sin(index * 0.3)
        + 20 * np.cos(index * 0.11)
    )

    high = (
        close
        + 5
        + 3 * np.abs(np.sin(index * 0.2))
    )

    low = (
        close
        - 5
        - 3 * np.abs(np.cos(index * 0.17))
    )

    volume = (
        10
        + index % 17
        + 4 * np.abs(np.sin(index * 0.4))
    )

    timestamps = (
        1794000000000
        + index * 5 * 60 * 1000
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })


def compare_indicator(name, actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: Output length mismatch."
        )

    if not np.array_equal(
        np.isnan(actual),
        np.isnan(expected)
    ):
        raise AssertionError(
            f"{name}: Warm-up or NaN position mismatch."
        )

    valid = np.isfinite(expected)

    if not np.allclose(
        actual[valid],
        expected[valid],
        rtol=1e-10,
        atol=TOLERANCE
    ):
        difference = np.max(
            np.abs(actual[valid] - expected[valid])
        )

        raise AssertionError(
            f"{name}: Calculation mismatch. "
            f"Maximum difference: {difference}"
        )

    print(f"{name}: PASS")


def main():
    print("INDEPENDENT INDICATOR UNIT TEST")
    print("================================")
    print(f"Test Candles: {TEST_CANDLES}")
    print("Data Source: Synthetic Test Data")
    print("Archive Modification: DISABLED")

    data = build_test_data()

    result = calculate_all_indicators(data)

    tests = {
        "EMA 20": (
            result["ema_20"],
            manual_ema(data["close"], 20)
        ),
        "EMA 50": (
            result["ema_50"],
            manual_ema(data["close"], 50)
        ),
        "EMA 200": (
            result["ema_200"],
            manual_ema(data["close"], 200)
        ),
        "RSI 14": (
            result["rsi_14"],
            manual_rsi(data["close"], 14)
        ),
        "MACD Line": (
            result["macd"],
            manual_macd(data["close"])[0]
        ),
        "MACD Signal": (
            result["macd_signal"],
            manual_macd(data["close"])[1]
        ),
        "MACD Histogram": (
            result["macd_histogram"],
            manual_macd(data["close"])[2]
        ),
        "ATR 14": (
            result["atr_14"],
            manual_atr(data, 14)
        ),
        "ADX 14": (
            result["adx_14"],
            manual_adx(data, 14)
        ),
        "Average Volume 20": (
            result["average_volume_20"],
            manual_volume_analysis(data, 20)[0]
        ),
        "Volume Ratio": (
            result["volume_ratio"],
            manual_volume_analysis(data, 20)[1]
        ),
        "Daily VWAP": (
            result["vwap"],
            manual_vwap(data)
        )
    }

    all_passed = True

    print("\nINDIVIDUAL TEST RESULTS")
    print("-----------------------")

    for name, values in tests.items():
        try:
            compare_indicator(
                name,
                values[0],
                values[1]
            )

        except Exception as error:
            all_passed = False
            print(f"{name}: FAIL")
            print(f"ERROR: {error}")

    print("\nFINAL UNIT TEST REPORT")
    print("======================")

    print(
        "OVERALL UNIT TEST STATUS: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print("ARCHIVE MODIFIED: NO")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
