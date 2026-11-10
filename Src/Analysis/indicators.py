
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

PRICE_FIELDS = [
    "open",
    "high",
    "low",
    "close"
]


def validate_candles(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if data.empty:
        raise ValueError("Candle data is empty.")

    result = data.copy()

    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    if result[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Invalid or missing candle values.")

    numeric_values = result[
        REQUIRED_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Non-finite candle values detected.")

    if (result["timestamp"] <= 0).any():
        raise ValueError("Invalid timestamp.")

    if (
        result["timestamp"] % 1 != 0
    ).any():
        raise ValueError("Timestamp must be an integer.")

    result["timestamp"] = result[
        "timestamp"
    ].astype("int64")

    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    if (result[PRICE_FIELDS] <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    if (
        result["high"]
        < result[["open", "close", "low"]].max(axis=1)
    ).any():
        raise ValueError("Invalid high price.")

    if (
        result["low"]
        > result[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Invalid low price.")

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return result


def calculate_ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

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

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100
    )

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50
    )

    return rsi


def calculate_macd(series):
    ema_12 = calculate_ema(series, 12)
    ema_26 = calculate_ema(series, 26)

    macd_line = ema_12 - ema_26

    signal_line = macd_line.ewm(
        span=9,
        adjust=False,
        min_periods=9
    ).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


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
    upward_move = data["high"].diff()

    downward_move = -data["low"].diff()

    positive_dm = pd.Series(
        np.where(
            (upward_move > downward_move)
            & (upward_move > 0),
            upward_move,
            0
        ),
        index=data.index
    )

    negative_dm = pd.Series(
        np.where(
            (downward_move > upward_move)
            & (downward_move > 0),
            downward_move,
            0
        ),
        index=data.index
    )

    atr = calculate_atr(data, period)

    positive_smoothed = positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    negative_smoothed = negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    positive_di = 100 * positive_smoothed / atr
    negative_di = 100 * negative_smoothed / atr

    denominator = (
        positive_di + negative_di
    ).replace(0, np.nan)

    dx = (
        100
        * (positive_di - negative_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


def calculate_vwap(data):
    typical_price = (
        data["high"]
        + data["low"]
        + data["close"]
    ) / 3

    utc_days = pd.to_datetime(
        data["timestamp"],
        unit="ms",
        utc=True
    ).dt.floor("D")

    cumulative_volume = (
        data["volume"]
        .groupby(utc_days, sort=False)
        .cumsum()
    )

    cumulative_value = (
        (typical_price * data["volume"])
        .groupby(utc_days, sort=False)
        .cumsum()
    )

    return (
        cumulative_value
        / cumulative_volume.replace(0, np.nan)
    )


def calculate_volume_analysis(data, period=20):
    average_volume = data["volume"].rolling(
        window=period,
        min_periods=period
    ).mean()

    volume_ratio = (
        data["volume"]
        / average_volume.replace(0, np.nan)
    )

    return average_volume, volume_ratio


def calculate_all_indicators(data):
    result = validate_candles(data)

    result["ema_20"] = calculate_ema(
        result["close"], 20
    )

    result["ema_50"] = calculate_ema(
        result["close"], 50
    )

    result["ema_200"] = calculate_ema(
        result["close"], 200
    )

    result["rsi_14"] = calculate_rsi(
        result["close"], 14
    )

    (
        result["macd"],
        result["macd_signal"],
        result["macd_histogram"]
    ) = calculate_macd(result["close"])

    result["atr_14"] = calculate_atr(
        result, 14
    )

    result["adx_14"] = calculate_adx(
        result, 14
    )

    result["vwap"] = calculate_vwap(result)

    (
        result["average_volume_20"],
        result["volume_ratio"]
    ) = calculate_volume_analysis(result, 20)

    return result
