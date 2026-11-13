
from Src.Engines.Trendmomentum.data_loader import load_all_timeframes
from Src.Engines.Trendmomentum.multi_timeframe_engine import (
    generate_multi_timeframe_signals,
)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_counts(title, series):
    print(f"\n{title}")
    print("-" * 70)

    counts = series.value_counts(dropna=False)

    for label, count in counts.items():
        percentage = count / len(series) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")


def main():
    print_section("MULTI-TIMEFRAME ENGINE AUDIT")

    datasets = load_all_timeframes()

    for timeframe in ["1D", "4H", "1H"]:
        print(
            f"{timeframe} candles: "
            f"{len(datasets[timeframe])}"
        )

    result = generate_multi_timeframe_signals(
        datasets,
        breakout_lookback=20,
        pullback_atr_tolerance=0.25,
    )

    if result.empty:
        raise AssertionError("Engine returned an empty result.")

    required_columns = [
        "timestamp",
        "signal_time",
        "signal",
        "signal_type",
        "1h_signal",
        "1h_signal_type",
        "1h_signal_reason",
        "1d_market_regime",
        "4h_market_regime",
        "1d_trend_alignment",
        "4h_trend_alignment",
        "1d_trend_strength",
        "4h_trend_strength",
        "daily_confirmation",
        "four_hour_confirmation",
        "4h_close_time",
        "1d_close_time",
        "4h_data_age_hours",
        "1d_data_age_hours",
    ]

    missing = [
        column for column in required_columns
        if column not in result.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing output columns: {missing}"
        )

    allowed_signals = {"LONG", "SHORT", "WAIT"}

    invalid_signals = (
        set(result["signal"].dropna().unique())
        - allowed_signals
    )

    if invalid_signals:
        raise AssertionError(
            f"Invalid final signals: {invalid_signals}"
        )

    invalid_hourly_signals = (
        set(result["1h_signal"].dropna().unique())
        - {"LONG", "SHORT", "WAIT"}
    )

    if invalid_hourly_signals:
        raise AssertionError(
            f"Invalid 1H signals: {invalid_hourly_signals}"
        )

    if result["timestamp"].duplicated().any():
        raise AssertionError(
            "Duplicate hourly timestamps found."
        )

    if (result["4h_close_time"] > result["signal_time"]).any():
        raise AssertionError("Future 4H data detected.")

    if (result["1d_close_time"] > result["signal_time"]).any():
        raise AssertionError("Future 1D data detected.")

    if (result["4h_data_age_hours"] < 0).any():
        raise AssertionError("Negative 4H data age detected.")

    if (result["1d_data_age_hours"] < 0).any():
        raise AssertionError("Negative 1D data age detected.")

    print_section("1. RAW 1H SIGNALS")

    print_counts(
        "Hourly signal distribution",
        result["1h_signal"]
    )

    print_counts(
        "Hourly signal types",
        result["1h_signal_type"]
    )

    print_section("2. DAILY MARKET REGIME")

    print_counts(
        "Daily regime distribution",
        result["1d_market_regime"]
    )

    print_counts(
        "Daily trend alignment",
        result["1d_trend_alignment"]
    )

    print_counts(
        "Daily trend strength",
        result["1d_trend_strength"]
    )

    print_section("3. FOUR-HOUR MARKET REGIME")

    print_counts(
        "4H regime distribution",
        result["4h_market_regime"]
    )

    print_counts(
        "4H trend alignment",
        result["4h_trend_alignment"]
    )

    print_counts(
        "4H trend strength",
        result["4h_trend_strength"]
    )

    print_section("4. HIGHER TIMEFRAME CONFIRMATIONS")

    print_counts(
        "Daily confirmation",
        result["daily_confirmation"]
    )

    print_counts(
        "4H confirmation",
        result["four_hour_confirmation"]
    )

    print("\nConfirmation combinations")
    print("-" * 70)

    combinations = (
        result.groupby(
            [
                "daily_confirmation",
                "four_hour_confirmation",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    for _, row in combinations.iterrows():
        print(
            f"Daily={row['daily_confirmation']} | "
            f"4H={row['four_hour_confirmation']} | "
            f"Count={row['count']}"
        )

    print_section("5. SIGNAL FILTER AUDIT")

    daily_bullish = (
        result["1d_market_regime"].eq("BULLISH_TREND")
        & result["1d_trend_alignment"].eq("BULLISH_CONFIRMED")
    )

    daily_bearish = (
        result["1d_market_regime"].eq("BEARISH_TREND")
        & result["1d_trend_alignment"].eq("BEARISH_CONFIRMED")
    )

    four_hour_bullish = (
        result["4h_market_regime"].eq("BULLISH_TREND")
        & result["4h_trend_alignment"].eq("BULLISH_CONFIRMED")
        & result["4h_trend_strength"].isin(
            ["STRONG", "MODERATE"]
        )
    )

    four_hour_bearish = (
        result["4h_market_regime"].eq("BEARISH_TREND")
        & result["4h_trend_alignment"].eq("BEARISH_CONFIRMED")
        & result["4h_trend_strength"].isin(
            ["STRONG", "MODERATE"]
        )
    )

    raw_long = result["1h_signal"].eq("LONG")
    raw_short = result["1h_signal"].eq("SHORT")

    daily_long_candidates = raw_long & daily_bullish
    daily_short_candidates = raw_short & daily_bearish

    confirmed_long = (
        daily_long_candidates & four_hour_bullish
    )

    confirmed_short = (
        daily_short_candidates & four_hour_bearish
    )

    audit_counts = {
        "Raw 1H LONG": raw_long.sum(),
        "Raw 1H SHORT": raw_short.sum(),
        "LONG passing daily filter": daily_long_candidates.sum(),
        "SHORT passing daily filter": daily_short_candidates.sum(),
        "LONG passing daily and 4H filters": confirmed_long.sum(),
        "SHORT passing daily and 4H filters": confirmed_short.sum(),
        "Final LONG": result["signal"].eq("LONG").sum(),
        "Final SHORT": result["signal"].eq("SHORT").sum(),
        "Final WAIT": result["signal"].eq("WAIT").sum(),
    }

    for label, count in audit_counts.items():
        print(f"{label}: {count}")

    print_section("6. LATEST ENGINE OUTPUT")

    latest = result.iloc[-1]

    fields = [
        "timestamp",
        "signal_time",
        "1h_signal",
        "1h_signal_type",
        "1h_signal_reason",
        "1d_market_regime",
        "1d_trend_alignment",
        "4h_market_regime",
        "4h_trend_alignment",
        "daily_confirmation",
        "four_hour_confirmation",
        "signal",
        "signal_type",
        "signal_reason",
        "4h_data_age_hours",
        "1d_data_age_hours",
    ]

    for field in fields:
        print(f"{field}: {latest[field]}")

    print_section("7. DATA ALIGNMENT VALIDATION")

    print(f"Aligned rows: {len(result)}")
    print(
        "Future 4H matches: "
        f"{(result['4h_close_time'] > result['signal_time']).sum()}"
    )
    print(
        "Future 1D matches: "
        f"{(result['1d_close_time'] > result['signal_time']).sum()}"
    )
    print(
        "Duplicate hourly timestamps: "
        f"{result['timestamp'].duplicated().sum()}"
    )

    print_section("8. FINAL RESULT")

    print("STATUS: PASS")
    print("RESULT: PASS")
    print("NOTE: PASS confirms execution and data checks only.")
    print("NOTE: It does not establish trading profitability.")


if __name__ == "__main__":
    main()
