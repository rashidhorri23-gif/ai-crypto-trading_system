
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def load_market_data(file_path):
    """Load and validate historical OHLCV data."""

    df = pd.read_csv(file_path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    numeric_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid market candles found."
        )

    if df["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found."
        )

    if (df["high"] < df["low"]).any():
        raise ValueError(
            "Invalid candle: high is below low."
        )

    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            "Invalid OHLC candle values found."
        )

    if (df["volume"] < 0).any():
        raise ValueError(
            "Negative volume values found."
        )

    return df


def calculate_ema(series, period):
    """Calculate exponential moving average."""

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_rsi(series, period=14):
    """Calculate RSI using Wilder smoothing."""

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = (
        avg_gain / avg_loss.replace(0, np.nan)
    )

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100,
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50,
    )

    return rsi


def calculate_atr(df, period=14):
    """Calculate Average True Range."""

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_adx(df, period=14):
    """Calculate ADX and directional indicators."""

    high_difference = df["high"].diff()
    low_difference = -df["low"].diff()

    positive_dm = pd.Series(
        np.where(
            (high_difference > low_difference)
            & (high_difference > 0),
            high_difference,
            0.0,
        ),
        index=df.index,
    )

    negative_dm = pd.Series(
        np.where(
            (low_difference > high_difference)
            & (low_difference > 0),
            low_difference,
            0.0,
        ),
        index=df.index,
    )

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_smooth = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    positive_smoothed = positive_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    negative_smoothed = negative_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100 * positive_smoothed / atr_smooth
    )

    minus_di = (
        100 * negative_smoothed / atr_smooth
    )

    denominator = (
        plus_di + minus_di
    ).replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return adx, plus_di, minus_di


def calculate_market_structure(df, swing_window=3):
    """Calculate confirmed swing points without lookahead."""

    result = df.copy()

    window = (2 * swing_window) + 1

    candidate_high = (
        result["high"]
        .shift(swing_window)
        .rolling(
            window=window,
            min_periods=window,
        )
        .max()
    )

    candidate_low = (
        result["low"]
        .shift(swing_window)
        .rolling(
            window=window,
            min_periods=window,
        )
        .min()
    )

    pivot_high = (
        result["high"].shift(swing_window)
        == candidate_high
    )

    pivot_low = (
        result["low"].shift(swing_window)
        == candidate_low
    )

    result["swing_high"] = (
        result["high"]
        .shift(swing_window)
        .where(pivot_high)
    )

    result["swing_low"] = (
        result["low"]
        .shift(swing_window)
        .where(pivot_low)
    )

    result["last_swing_high"] = (
        result["swing_high"].ffill()
    )

    result["last_swing_low"] = (
        result["swing_low"].ffill()
    )

    previous_high = (
        result["last_swing_high"].shift(1)
    )

    previous_low = (
        result["last_swing_low"].shift(1)
    )

    result["bos_bullish"] = (
        (result["close"] > previous_high)
        & (previous_high.notna())
    )

    result["bos_bearish"] = (
        (result["close"] < previous_low)
        & (previous_low.notna())
    )

    bullish_break = result["bos_bullish"]
    bearish_break = result["bos_bearish"]

    result["market_structure"] = "NEUTRAL"

    result.loc[
        bullish_break,
        "market_structure",
    ] = "BULLISH_BOS"

    result.loc[
        bearish_break,
        "market_structure",
    ] = "BEARISH_BOS"

    result["last_bullish_bos"] = (
        result["close"].where(bullish_break).ffill()
    )

    result["last_bearish_bos"] = (
        result["close"].where(bearish_break).ffill()
    )

    return result


def build_market_features(df):
    """Build all technical features for historical candles."""

    result = df.copy()

    result["ema_20"] = calculate_ema(
        result["close"],
        20,
    )

    result["ema_50"] = calculate_ema(
        result["close"],
        50,
    )

    result["ema_200"] = calculate_ema(
        result["close"],
        200,
    )

    result["rsi_14"] = calculate_rsi(
        result["close"],
        14,
    )

    result["atr_14"] = calculate_atr(
        result,
        14,
    )

    (
        result["adx_14"],
        result["plus_di"],
        result["minus_di"],
    ) = calculate_adx(
        result,
        14,
    )

    result["volume_sma_20"] = (
        result["volume"]
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    result["relative_volume"] = (
        result["volume"]
        / result["volume_sma_20"].replace(0, np.nan)
    )

    result = calculate_market_structure(
        result,
        swing_window=3,
    )

    result["support"] = (
        result["last_swing_low"]
    )

    result["resistance"] = (
        result["last_swing_high"]
    )

    result["ema_trend"] = "NEUTRAL"

    result.loc[
        (result["ema_50"] > result["ema_200"])
        & (result["close"] > result["ema_50"]),
        "ema_trend",
    ] = "BULLISH"

    result.loc[
        (result["ema_50"] < result["ema_200"])
        & (result["close"] < result["ema_50"]),
        "ema_trend",
    ] = "BEARISH"

    result["timestamp_utc"] = pd.to_datetime(
        result["timestamp"],
        unit="ms",
        utc=True,
    )

    return result


def prepare_market_features(file_path):
    """Load historical candles and calculate technical features."""

    market_data = load_market_data(
        file_path
    )

    return build_market_features(
        market_data
    )
