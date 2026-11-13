
import numpy as np
import pandas as pd


def analyze_trend(data):
    required = [
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "ema_200_slope",
        "rsi_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "market_regime"
    ]

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()

    numeric_columns = [
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "ema_200_slope",
        "rsi_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14"
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result["trend_direction"] = "NEUTRAL"
    result["trend_strength"] = "UNDEFINED"
    result["momentum_direction"] = "NEUTRAL"
    result["trend_alignment"] = "UNCONFIRMED"

    valid = (
        result[numeric_columns].notna().all(axis=1)
        & np.isfinite(result[numeric_columns]).all(axis=1)
        & (result["close"] > 0)
        & (result["ema_20"] > 0)
        & (result["ema_50"] > 0)
        & (result["ema_200"] > 0)
        & result["rsi_14"].between(0, 100)
        & result["adx_14"].between(0, 100)
        & result["plus_di_14"].between(0, 100)
        & result["minus_di_14"].between(0, 100)
    )

    bullish = (
        valid
        & (result["ema_20"] > result["ema_50"])
        & (result["ema_50"] > result["ema_200"])
        & (result["ema_200_slope"] > 0)
    )

    bearish = (
        valid
        & (result["ema_20"] < result["ema_50"])
        & (result["ema_50"] < result["ema_200"])
        & (result["ema_200_slope"] < 0)
    )

    result.loc[bullish, "trend_direction"] = "BULLISH"
    result.loc[bearish, "trend_direction"] = "BEARISH"

    strong = valid & (result["adx_14"] >= 25)

    moderate = (
        valid
        & (result["adx_14"] >= 20)
        & (result["adx_14"] < 25)
    )

    weak = valid & (result["adx_14"] < 20)

    result.loc[strong, "trend_strength"] = "STRONG"
    result.loc[moderate, "trend_strength"] = "MODERATE"
    result.loc[weak, "trend_strength"] = "WEAK"

    bullish_momentum = (
        valid
        & (result["plus_di_14"] > result["minus_di_14"])
        & (result["rsi_14"] > 50)
    )

    bearish_momentum = (
        valid
        & (result["minus_di_14"] > result["plus_di_14"])
        & (result["rsi_14"] < 50)
    )

    result.loc[bullish_momentum, "momentum_direction"] = "BULLISH"
    result.loc[bearish_momentum, "momentum_direction"] = "BEARISH"

    bullish_alignment = (
        bullish
        & bullish_momentum
        & result["market_regime"].eq("BULLISH_TREND")
    )

    bearish_alignment = (
        bearish
        & bearish_momentum
        & result["market_regime"].eq("BEARISH_TREND")
    )

    result.loc[
        bullish_alignment,
        "trend_alignment"
    ] = "BULLISH_CONFIRMED"

    result.loc[
        bearish_alignment,
        "trend_alignment"
    ] = "BEARISH_CONFIRMED"

    return result
