
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


def prepare_timeframe(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Timeframe data must be a pandas DataFrame.")

    result = calculate_indicators(data)
    result = classify_market_regime(result)
    result = analyze_trend(result)

    return result


def validate_timeframe_data(data, timeframe):
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

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

    result = data.copy()

    result["timestamp"] = pd.to_numeric(
        result["timestamp"],
        errors="coerce"
    )

    if result["timestamp"].isna().any():
        raise ValueError(f"{timeframe} has invalid timestamps.")

    if result["timestamp"].duplicated().any():
        raise ValueError(f"{timeframe} has duplicate timestamps.")

    result = result.sort_values("timestamp").reset_index(drop=True)

    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{timeframe} timestamps are not ordered.")

    interval_ms = TIMEFRAME_MS[timeframe]

    if (result["timestamp"] % interval_ms != 0).any():
        raise ValueError(f"{timeframe} timestamps are misaligned.")

    return result


def build_multi_timeframe_dataset(datasets):
    required_timeframes = ["1D", "4H", "1H"]

    missing = [
        timeframe for timeframe in required_timeframes
        if timeframe not in datasets
    ]

    if missing:
        raise ValueError(
            f"Missing timeframe datasets: {missing}"
        )

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
        raise ValueError(
            "No valid multi-timeframe rows could be aligned."
        )

    if (
        result["4h_close_time"] > result["signal_time"]
    ).any():
        raise RuntimeError("Future 4H data detected.")

    if (
        result["1d_close_time"] > result["signal_time"]
    ).any():
        raise RuntimeError("Future 1D data detected.")

    return result


def generate_multi_timeframe_signals(
    datasets,
    breakout_lookback=20,
    pullback_atr_tolerance=0.25,
):
    aligned = build_multi_timeframe_dataset(datasets)

    hourly = prepared_hourly_signals(aligned)

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

    long_allowed = daily_bullish & four_hour_bullish

    short_allowed = daily_bearish & four_hour_bearish

    result = aligned.copy()

    result["signal"] = "WAIT"
    result["signal_type"] = "NONE"
    result["signal_reason"] = "Higher timeframe confirmation missing"

    long_signal = (
        long_allowed
        & hourly["signal"].eq("LONG")
    )

    short_signal = (
        short_allowed
        & hourly["signal"].eq("SHORT")
    )

    result.loc[long_signal, "signal"] = "LONG"
    result.loc[long_signal, "signal_type"] = (
        hourly.loc[long_signal, "signal_type"]
    )
    result.loc[long_signal, "signal_reason"] = (
        "1D and 4H bullish confirmation with 1H entry trigger"
    )

    result.loc[short_signal, "signal"] = "SHORT"
    result.loc[short_signal, "signal_type"] = (
        hourly.loc[short_signal, "signal_type"]
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

    return result


def prepared_hourly_signals(aligned):
    hourly = aligned.copy()

    hourly = generate_signals(hourly)

    return hourly
