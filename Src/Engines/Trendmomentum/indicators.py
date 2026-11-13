
import numpy as np
import pandas as pd


def validate_period(period):
    if not isinstance(period, int) or isinstance(period, bool):
        raise TypeError("Period must be an integer.")

    if period <= 0:
        raise ValueError("Period must be positive.")


def validate_ohlcv(data):
    required = ["open", "high", "low", "close", "volume"]

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data.empty:
        raise ValueError("Input data is empty.")

    result = data.copy()

    for column in required:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    values = result[required].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("OHLCV contains missing or non-finite values.")

    price_columns = ["open", "high", "low", "close"]

    if (result[price_columns] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")

    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    if (result["high"] < result["low"]).any():
        raise ValueError("Invalid candle: high is below low.")

    if (
        (result["open"] > result["high"])
        | (result["open"] < result["low"])
    ).any():
        raise ValueError("Candle open is outside high-low range.")

    if (
        (result["close"] > result["high"])
        | (result["close"] < result["low"])
    ).any():
        raise ValueError("Candle close is outside high-low range.")

    return result


def calculate_ema(data, period, column="close"):
    validate_period(period)

    if column not in data.columns:
        raise ValueError(f"Missing column: {column}")

    return data[column].ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_rsi(data, period=14):
    validate_period(period)

    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50
    )

    return rsi


def calculate_atr(data, period=14):
    validate_period(period)

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_adx(data, period=14):
    validate_period(period)

    high_diff = data["high"].diff()
    low_diff = -data["low"].diff()

    positive_dm = pd.Series(
        np.where(
            (high_diff > low_diff) & (high_diff > 0),
            high_diff,
            0.0
        ),
        index=data.index
    )

    negative_dm = pd.Series(
        np.where(
            (low_diff > high_diff) & (low_diff > 0),
            low_diff,
            0.0
        ),
        index=data.index
    )

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    smoothed_positive_dm = positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    smoothed_negative_dm = negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    positive_di = (
        100 * smoothed_positive_dm / atr.replace(0, np.nan)
    )

    negative_di = (
        100 * smoothed_negative_dm / atr.replace(0, np.nan)
    )

    di_sum = positive_di + negative_di

    dx = (
        100 * (positive_di - negative_di).abs()
        / di_sum.replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return pd.DataFrame(
        {
            "plus_di": positive_di,
            "minus_di": negative_di,
            "adx": adx
        },
        index=data.index
    )


def calculate_indicators(data):
    result = validate_ohlcv(data)

    result["ema_20"] = calculate_ema(result, 20)
    result["ema_50"] = calculate_ema(result, 50)
    result["ema_200"] = calculate_ema(result, 200)

    result["rsi_14"] = calculate_rsi(result, 14)
    result["atr_14"] = calculate_atr(result, 14)

    adx_data = calculate_adx(result, 14)

    result["plus_di_14"] = adx_data["plus_di"]
    result["minus_di_14"] = adx_data["minus_di"]
    result["adx_14"] = adx_data["adx"]

    result["ema_200_slope"] = (
        result["ema_200"] - result["ema_200"].shift(5)
    )

    return result
