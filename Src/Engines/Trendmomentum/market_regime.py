
import pandas as pd


def classify_market_regime(data):
    required = [
        "close",
        "ema_50",
        "ema_200",
        "ema_200_slope",
        "adx_14",
        "plus_di_14",
        "minus_di_14"
    ]

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()

    result["market_regime"] = "UNDEFINED"

    valid = result[required].notna().all(axis=1)

    strong_trend = result["adx_14"] >= 20

    bullish = (
        valid
        & strong_trend
        & (result["close"] > result["ema_200"])
        & (result["ema_50"] > result["ema_200"])
        & (result["ema_200_slope"] > 0)
        & (result["plus_di_14"] > result["minus_di_14"])
    )

    bearish = (
        valid
        & strong_trend
        & (result["close"] < result["ema_200"])
        & (result["ema_50"] < result["ema_200"])
        & (result["ema_200_slope"] < 0)
        & (result["minus_di_14"] > result["plus_di_14"])
    )

    ranging = (
        valid
        & ~strong_trend
    )

    result.loc[bullish, "market_regime"] = "BULLISH_TREND"
    result.loc[bearish, "market_regime"] = "BEARISH_TREND"
    result.loc[ranging, "market_regime"] = "RANGE"

    return result

