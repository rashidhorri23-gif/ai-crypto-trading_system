
import numpy as np
import pandas as pd


def _validate_ohlcv(data):
    required = ["open", "high", "low", "close", "volume"]

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}"
        )

    if data.empty:
        raise ValueError("Input data is empty.")

    result = data.copy()

    for column in required:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    if result[required].isnull().any().any():
        raise ValueError("OHLCV data contains invalid or missing values.")

    if (result["high"] < result["low"]).any():
        raise ValueError("Invalid candle: high is below low.")

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
    if period <= 0:
        raise ValueError("RSI period must be positive.")

    delta = data["close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    relative_strength = average_gain / average_loss

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss == 0),
        50
    )

    return rsi


def calculate_atr(data, period=14):
    if period <= 0:
        raise ValueError("ATR period must be positive.")

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
    if period <= 0:
        raise ValueError("ADX period must be positive.")

    high_difference = data["high"].diff()
    low_difference = -data["low"].diff()

    positive_dm = pd.Series(
        np.where(
            (high_difference > low_difference)
            & (high_difference > 0),
            high_difference,
            0.0
        ),
        index=data.index
    )

    negative_dm = pd.Series(
        np.where(
            (low_difference > high_difference)
            & (low_difference > 0),
            low_difference,
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

    smoothed_tr = true_range.ewm(
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

    positive_di = 100 * smoothed_positive_dm / smoothed_tr
    negative_di = 100 * smoothed_negative_dm / smoothed_tr

    di_sum = positive_di + negative_di
    di_difference = (positive_di - negative_di).abs()

    dx = 100 * di_difference / di_sum.replace(0, np.nan)

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
    result = _validate_ohlcv(data)

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

