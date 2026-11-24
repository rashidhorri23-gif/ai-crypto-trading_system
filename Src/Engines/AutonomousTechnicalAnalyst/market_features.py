
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


def validate_period(period, name="period"):
    if (
        not isinstance(period, (int, np.integer))
        or isinstance(period, (bool, np.bool_))
        or period <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def validate_ohlcv_data(df):
    """Validate OHLCV values without silently removing candles."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Market data must be a pandas DataFrame.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = df.copy()

    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if result[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing or non-numeric OHLCV values found."
        )

    values = result[REQUIRED_COLUMNS].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite market data values found."
        )

    if (result["timestamp"] <= 0).any():
        raise ValueError(
            "Invalid timestamps found."
        )

    if (
        result["timestamp"]
        != np.floor(result["timestamp"])
    ).any():
        raise ValueError(
            "Timestamps must be integer Unix milliseconds."
        )

    if result["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found."
        )

    if (
        result["open"] <= 0
    ).any() or (
        result["high"] <= 0
    ).any() or (
        result["low"] <= 0
    ).any() or (
        result["close"] <= 0
    ).any():
        raise ValueError(
            "OHLC prices must be greater than zero."
        )

    if (result["volume"] < 0).any():
        raise ValueError(
            "Negative volume values found."
        )

    if (result["high"] < result["low"]).any():
        raise ValueError(
            "Invalid candle: high is below low."
        )

    invalid_ohlc = (
        (result["high"] < result["open"])
        | (result["high"] < result["close"])
        | (result["low"] > result["open"])
        | (result["low"] > result["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            "Invalid OHLC candle values found."
        )

    return result


def load_market_data(file_path):
    """Load and validate historical OHLCV data."""

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "Market data file is empty."
        ) from error

    if df.empty:
        raise ValueError(
            "No market candles found."
        )

    df = validate_ohlcv_data(df)

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def calculate_ema(series, period):
    """Calculate exponential moving average."""

    validate_period(period)

    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_rsi(series, period=14):
    """Calculate RSI using Wilder-style smoothing."""

    validate_period(period)

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

    validate_period(period)

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

    validate_period(period)

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
        dtype=float,
    )

    negative_dm = pd.Series(
        np.where(
            (low_difference > high_difference)
            & (low_difference > 0),
            low_difference,
            0.0,
        ),
        index=df.index,
        dtype=float,
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

    validate_period(
        swing_window,
        "swing_window",
    )

    result = df.copy()

    window = (2 * swing_window) + 1

    candidate_high = (
        result["high"]
        .rolling(
            window=window,
            min_periods=window,
        )
        .max()
    )

    candidate_low = (
        result["low"]
        .rolling(
            window=window,
            min_periods=window,
        )
        .min()
    )

    # A swing is recorded only after swing_window
    # candles have closed to its right.

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
        & previous_high.notna()
    )

    result["bos_bearish"] = (
        (result["close"] < previous_low)
        & previous_low.notna()
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
        result["close"]
        .where(bullish_break)
        .ffill()
    )

    result["last_bearish_bos"] = (
        result["close"]
        .where(bearish_break)
        .ffill()
    )

    return result


def build_market_features(df):
    """Build technical features using current and past candles only."""

    result = validate_ohlcv_data(df)

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

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

    # Bollinger Bands and MACD provide complementary volatility/momentum context.
    result["bb_middle"] = result["close"].rolling(20, min_periods=20).mean()
    bb_std = result["close"].rolling(20, min_periods=20).std(ddof=0)
    result["bb_upper"] = result["bb_middle"] + (2.0 * bb_std)
    result["bb_lower"] = result["bb_middle"] - (2.0 * bb_std)

    macd_fast = calculate_ema(result["close"], 12)
    macd_slow = calculate_ema(result["close"], 26)
    result["macd"] = macd_fast - macd_slow
    result["macd_signal"] = calculate_ema(result["macd"], 9)
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    # UTC-session anchored VWAP. Zero-volume sessions remain unavailable.
    typical_price = (result["high"] + result["low"] + result["close"]) / 3.0
    session = pd.to_datetime(result["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    cumulative_pv = (typical_price * result["volume"]).groupby(session).cumsum()
    cumulative_volume = result["volume"].groupby(session).cumsum().replace(0, np.nan)
    result["vwap_utc"] = cumulative_pv / cumulative_volume

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

    try:
        result["timestamp_utc"] = pd.to_datetime(
            result["timestamp"],
            unit="ms",
            utc=True,
            errors="raise",
        )
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(
            "Could not convert timestamps to UTC."
        ) from error

    return result


def prepare_market_features(file_path):
    """Load historical candles and calculate technical features."""

    market_data = load_market_data(file_path)

    return build_market_features(market_data)
