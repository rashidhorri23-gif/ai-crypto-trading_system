
import numpy as np
import pandas as pd

from Src.Engines.Trendmomentum.indicators import calculate_indicators
from Src.Engines.Trendmomentum.market_regime import classify_market_regime
from Src.Engines.Trendmomentum.trend_analysis import analyze_trend
from Src.Engines.Trendmomentum.signal_generator import generate_signals


TIMEFRAME_MS = {
    "1H": 3_600_000,
    "4H": 14_400_000,
    "1D": 86_400_000,
}


def validate_timeframe_data(data, timeframe):
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Timeframe data must be a pandas DataFrame.")

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{timeframe} data is missing columns: {missing}"
        )

    result = data[required].copy()

    for column in required:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    values = result[required].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{timeframe} contains invalid OHLCV values.")

    result["timestamp"] = result["timestamp"].astype("int64")

    if result["timestamp"].duplicated().any():
        raise ValueError(f"{timeframe} contains duplicate timestamps.")

    result = result.sort_values("timestamp").reset_index(drop=True)

    interval_ms = TIMEFRAME_MS[timeframe]

    if (result["timestamp"] % interval_ms != 0).any():
        raise ValueError(f"{timeframe} timestamps are misaligned.")

    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{timeframe} contains non-positive prices.")

    if (result["volume"] < 0).any():
        raise ValueError(f"{timeframe} contains negative volume.")

    if (result["high"] < result[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError(f"{timeframe} contains invalid high values.")

    if (result["low"] > result[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError(f"{timeframe} contains invalid low values.")

    return result


def prepare_timeframe(data):
    result = calculate_indicators(data)
    result = classify_market_regime(result)
    result = analyze_trend(result)

    return result


def build_multi_timeframe_dataset(datasets):
    required_timeframes = ["1D", "4H", "1H"]

    if not isinstance(datasets, dict):
        raise TypeError("Datasets must be a dictionary.")

    missing = [
        timeframe for timeframe in required_timeframes
        if timeframe not in datasets
    ]

    if missing:
        raise ValueError(f"Missing timeframe datasets: {missing}")

    prepared = {}

    for timeframe in required_timeframes:
        validated = validate_timeframe_data(
            datasets[timeframe],
            timeframe
        )

        prepared[timeframe] = prepare_timeframe(validated)

    hourly = prepared["1H"].copy()
    four_hour = prepared["4H"].copy()
    daily = prepared["1D"].copy()

    hourly["signal_time"] = (
        hourly["timestamp"] + TIMEFRAME_MS["1H"]
    )

    four_hour["4h_close_time"] = (
        four_hour["timestamp"] + TIMEFRAME_MS["4H"]
    )

    daily["1d_close_time"] = (
        daily["timestamp"] + TIMEFRAME_MS["1D"]
    )

    four_hour_columns = [
        "4h_close_time",
        "market_regime",
        "trend_direction",
        "trend_strength",
        "momentum_direction",
        "trend_alignment",
        "close",
        "adx_14",
        "rsi_14",
    ]

    daily_columns = [
        "1d_close_time",
        "market_regime",
        "trend_direction",
        "trend_strength",
        "momentum_direction",
        "trend_alignment",
        "close",
        "adx_14",
        "rsi_14",
    ]

    four_hour = four_hour[four_hour_columns].rename(
        columns={
            column: f"4h_{column}"
            for column in four_hour_columns
            if column != "4h_close_time"
        }
    )

    daily = daily[daily_columns].rename(
        columns={
            column: f"1d_{column}"
            for column in daily_columns
            if column != "1d_close_time"
        }
    )

    result = pd.merge_asof(
        hourly.sort_values("signal_time"),
        four_hour.sort_values("4h_close_time"),
        left_on="signal_time",
        right_on="4h_close_time",
        direction="backward",
        allow_exact_matches=True,
    )

    result = pd.merge_asof(
        result.sort_values("signal_time"),
        daily.sort_values("1d_close_time"),
        left_on="signal_time",
        right_on="1d_close_time",
        direction="backward",
        allow_exact_matches=True,
    )

    result = result.dropna(
        subset=["4h_close_time", "1d_close_time"]
    ).reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid multi-timeframe rows could be aligned.")

    if (result["4h_close_time"] > result["signal_time"]).any():
        raise RuntimeError("Future 4H data detected.")

    if (result["1d_close_time"] > result["signal_time"]).any():
        raise RuntimeError("Future 1D data detected.")

    return result


def generate_multi_timeframe_signals(
    datasets,
    breakout_lookback=20,
    pullback_atr_tolerance=0.25,
):
    if not isinstance(breakout_lookback, int) or isinstance(breakout_lookback, bool):
        raise TypeError("Breakout lookback must be an integer.")

    if breakout_lookback < 2:
        raise ValueError("Breakout lookback must be at least 2.")

    if not isinstance(pullback_atr_tolerance, (int, float)):
        raise TypeError("Pullback tolerance must be numeric.")

    if not np.isfinite(pullback_atr_tolerance) or pullback_atr_tolerance < 0:
        raise ValueError("Pullback tolerance must be finite and non-negative.")

    prepared = {}

    for timeframe in ["1D", "4H", "1H"]:
        validated = validate_timeframe_data(
            datasets[timeframe],
            timeframe
        )
        prepared[timeframe] = prepare_timeframe(validated)

    hourly_signals = generate_signals(
        prepared["1H"],
        breakout_lookback=breakout_lookback,
        pullback_atr_tolerance=pullback_atr_tolerance,
    )

    aligned = build_multi_timeframe_dataset(prepared)

    daily_bullish = (
        aligned["1d_market_regime"].eq("BULLISH_TREND")
        & aligned["1d_trend_alignment"].eq("BULLISH_CONFIRMED")
    )

    daily_bearish = (
        aligned["1d_market_regime"].eq("BEARISH_TREND")
        & aligned["1d_trend_alignment"].eq("BEARISH_CONFIRMED")
    )

    four_hour_bullish = (
        aligned["4h_market_regime"].eq("BULLISH_TREND")
        & aligned["4h_trend_alignment"].eq("BULLISH_CONFIRMED")
        & aligned["4h_trend_strength"].isin(["STRONG", "MODERATE"])
    )

    four_hour_bearish = (
        aligned["4h_market_regime"].eq("BEARISH_TREND")
        & aligned["4h_trend_alignment"].eq("BEARISH_CONFIRMED")
        & aligned["4h_trend_strength"].isin(["STRONG", "MODERATE"])
    )

    hourly_columns = [
        "timestamp",
        "signal",
        "signal_type",
        "signal_reason",
        "breakout_high",
        "breakout_low",
    ]

    hourly_signals = hourly_signals[hourly_columns].rename(
        columns={
            "timestamp": "1h_timestamp",
            "signal": "1h_signal",
            "signal_type": "1h_signal_type",
            "signal_reason": "1h_signal_reason",
            "breakout_high": "1h_breakout_high",
            "breakout_low": "1h_breakout_low",
        }
    )

    result = aligned.merge(
        hourly_signals,
        left_on="timestamp",
        right_on="1h_timestamp",
        how="left",
        validate="one_to_one",
    )

    long_allowed = daily_bullish & four_hour_bullish
    short_allowed = daily_bearish & four_hour_bearish

    long_signal = long_allowed & result["1h_signal"].eq("LONG")
    short_signal = short_allowed & result["1h_signal"].eq("SHORT")

    result["signal"] = "WAIT"
    result["signal_type"] = "NONE"
    result["signal_reason"] = "Higher timeframe confirmation missing"

    result.loc[long_signal, "signal"] = "LONG"
    result.loc[long_signal, "signal_type"] = (
        result.loc[long_signal, "1h_signal_type"]
    )
    result.loc[long_signal, "signal_reason"] = (
        "1D and 4H bullish confirmation with 1H entry trigger"
    )

    result.loc[short_signal, "signal"] = "SHORT"
    result.loc[short_signal, "signal_type"] = (
        result.loc[short_signal, "1h_signal_type"]
    )
    result.loc[short_signal, "signal_reason"] = (
        "1D and 4H bearish confirmation with 1H entry trigger"
    )

    result["daily_confirmation"] = np.select(
        [daily_bullish, daily_bearish],
        ["BULLISH", "BEARISH"],
        default="UNCONFIRMED",
    )

    result["four_hour_confirmation"] = np.select(
        [four_hour_bullish, four_hour_bearish],
        ["BULLISH", "BEARISH"],
        default="UNCONFIRMED",
    )

    result["1h_signal_time"] = (
        result["timestamp"] + TIMEFRAME_MS["1H"]
    )

    result["signal_time"] = result["1h_signal_time"]

    result["4h_data_age_hours"] = (
        result["signal_time"] - result["4h_close_time"]
    ) / 3_600_000

    result["1d_data_age_hours"] = (
        result["signal_time"] - result["1d_close_time"]
    ) / 3_600_000

    if (
        (result["4h_data_age_hours"] < 0).any()
        or (result["1d_data_age_hours"] < 0).any()
    ):
        raise RuntimeError("Future timeframe data detected.")

    return result
