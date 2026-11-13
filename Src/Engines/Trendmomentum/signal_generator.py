
import pandas as pd


def generate_signals(data):
    required = [
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "atr_14",
        "adx_14",
        "market_regime",
        "trend_direction",
        "trend_strength",
        "momentum_direction",
        "trend_alignment"
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

    result["signal"] = "WAIT"
    result["signal_reason"] = "Conditions not satisfied"

    valid = result[required].notna().all(axis=1)

    long_condition = (
        valid
        & (result["market_regime"] == "BULLISH_TREND")
        & (result["trend_direction"] == "BULLISH")
        & (result["trend_strength"].isin(["STRONG", "MODERATE"]))
        & (result["momentum_direction"] == "BULLISH")
        & (result["trend_alignment"] == "BULLISH_CONFIRMED")
        & (result["close"] > result["ema_20"])
        & (result["rsi_14"].between(50, 68))
        & (result["atr_14"] > 0)
    )

    short_condition = (
        valid
        & (result["market_regime"] == "BEARISH_TREND")
        & (result["trend_direction"] == "BEARISH")
        & (result["trend_strength"].isin(["STRONG", "MODERATE"]))
        & (result["momentum_direction"] == "BEARISH")
        & (result["trend_alignment"] == "BEARISH_CONFIRMED")
        & (result["close"] < result["ema_20"])
        & (result["rsi_14"].between(32, 50))
        & (result["atr_14"] > 0)
    )

    result.loc[long_condition, "signal"] = "LONG"
    result.loc[long_condition, "signal_reason"] = "Bullish trend and momentum confirmed"

    result.loc[short_condition, "signal"] = "SHORT"
    result.loc[short_condition, "signal_reason"] = "Bearish trend and momentum confirmed"

    return result
