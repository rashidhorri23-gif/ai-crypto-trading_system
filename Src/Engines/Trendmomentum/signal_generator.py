
import numpy as np
import pandas as pd


def generate_signals(data, breakout_lookback=20, pullback_atr_tolerance=0.25):
    required = [
        "open",
        "high",
        "low",
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

    if not isinstance(breakout_lookback, int) or isinstance(breakout_lookback, bool):
        raise TypeError("Breakout lookback must be an integer.")

    if breakout_lookback < 2:
        raise ValueError("Breakout lookback must be at least 2.")

    if not isinstance(pullback_atr_tolerance, (int, float)):
        raise TypeError("Pullback tolerance must be numeric.")

    if not np.isfinite(pullback_atr_tolerance) or pullback_atr_tolerance < 0:
        raise ValueError("Pullback tolerance must be finite and non-negative.")

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data.copy()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "atr_14",
        "adx_14"
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result["signal"] = "WAIT"
    result["signal_type"] = "NONE"
    result["signal_reason"] = "Conditions not satisfied"

    valid = (
        result[numeric_columns].notna().all(axis=1)
        & np.isfinite(result[numeric_columns]).all(axis=1)
        & (result["open"] > 0)
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
        & (result["atr_14"] > 0)
        & (result["adx_14"].between(0, 100))
        & (result["rsi_14"].between(0, 100))
        & (result["high"] >= result["low"])
        & (result["open"] <= result["high"])
        & (result["open"] >= result["low"])
        & (result["close"] <= result["high"])
        & (result["close"] >= result["low"])
    )

    previous_high = (
        result["high"]
        .rolling(
            window=breakout_lookback,
            min_periods=breakout_lookback
        )
        .max()
        .shift(1)
    )

    previous_low = (
        result["low"]
        .rolling(
            window=breakout_lookback,
            min_periods=breakout_lookback
        )
        .min()
        .shift(1)
    )

    result["breakout_high"] = previous_high
    result["breakout_low"] = previous_low

    bullish_trend = (
        valid
        & (result["market_regime"] == "BULLISH_TREND")
        & (result["trend_direction"] == "BULLISH")
        & (result["trend_strength"].isin(["STRONG", "MODERATE"]))
        & (result["momentum_direction"] == "BULLISH")
        & (result["trend_alignment"] == "BULLISH_CONFIRMED")
        & (result["rsi_14"].between(50, 68))
    )

    bearish_trend = (
        valid
        & (result["market_regime"] == "BEARISH_TREND")
        & (result["trend_direction"] == "BEARISH")
        & (result["trend_strength"].isin(["STRONG", "MODERATE"]))
        & (result["momentum_direction"] == "BEARISH")
        & (result["trend_alignment"] == "BEARISH_CONFIRMED")
        & (result["rsi_14"].between(32, 50))
    )

    bullish_pullback = (
        bullish_trend
        & (result["low"] <= (
            result["ema_20"] + result["atr_14"] * pullback_atr_tolerance
        ))
        & (result["close"] > result["ema_20"])
        & (result["close"] > result["open"])
    )

    bearish_pullback = (
        bearish_trend
        & (result["high"] >= (
            result["ema_20"] - result["atr_14"] * pullback_atr_tolerance
        ))
        & (result["close"] < result["ema_20"])
        & (result["close"] < result["open"])
    )

    bullish_breakout = (
        bullish_trend
        & previous_high.notna()
        & (result["close"] > previous_high)
        & (result["close"] > result["open"])
    )

    bearish_breakout = (
        bearish_trend
        & previous_low.notna()
        & (result["close"] < previous_low)
        & (result["close"] < result["open"])
    )

    result.loc[bullish_breakout, "signal"] = "LONG"
    result.loc[bullish_breakout, "signal_type"] = "BREAKOUT"
    result.loc[bullish_breakout, "signal_reason"] = "Bullish trend confirmed with upside breakout"

    result.loc[bearish_breakout, "signal"] = "SHORT"
    result.loc[bearish_breakout, "signal_type"] = "BREAKOUT"
    result.loc[bearish_breakout, "signal_reason"] = "Bearish trend confirmed with downside breakout"

    long_pullback_signal = (
        bullish_pullback
        & ~bullish_breakout
    )

    short_pullback_signal = (
        bearish_pullback
        & ~bearish_breakout
    )

    result.loc[long_pullback_signal, "signal"] = "LONG"
    result.loc[long_pullback_signal, "signal_type"] = "PULLBACK"
    result.loc[long_pullback_signal, "signal_reason"] = "Bullish trend confirmed after EMA20 pullback"

    result.loc[short_pullback_signal, "signal"] = "SHORT"
    result.loc[short_pullback_signal, "signal_type"] = "PULLBACK"
    result.loc[short_pullback_signal, "signal_reason"] = "Bearish trend confirmed after EMA20 pullback"

    return result
