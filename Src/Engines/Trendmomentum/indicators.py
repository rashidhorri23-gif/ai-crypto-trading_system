
import numpy as np
import pandas as pd


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
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result[required].isnull().any().any():
        raise ValueError("OHLCV data contains missing or invalid values.")

    if (result["high"] < result["low"]).any():
        raise ValueError("Invalid candle: high is below low.")

    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    return result


def calculate_ema(data, period, column="close"):
    if period <= 0:
        raise ValueError("EMA period must be positive.")

    return data[column].ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_rsi(data, period=14):
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

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    return rsi


def calculate_atr(data, period=14):
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

    positive_di = 100 * positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean() / atr

    negative_di = 100 * negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean() / atr

    di_sum = positive_di + negative_di
    dx = 100 * (positive_di - negative_di).abs() / di_sum.replace(0, np.nan)

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

