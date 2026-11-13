
from Src.Engines.Trendmomentum.data_loader import load_all_timeframes
from Src.Engines.Trendmomentum.multi_timeframe_engine import (
    generate_multi_timeframe_signals,
)


def main():
    print("=" * 65)
    print("MULTI-TIMEFRAME ENGINE TEST")
    print("=" * 65)

    datasets = load_all_timeframes()

    for timeframe, data in datasets.items():
        print(f"{timeframe} candles: {len(data)}")

    result = generate_multi_timeframe_signals(
        datasets,
        breakout_lookback=20,
        pullback_atr_tolerance=0.25,
    )

    if result.empty:
        raise AssertionError("Engine returned an empty result.")

    required_columns = [
        "signal",
        "signal_type",
        "signal_time",
        "4h_close_time",
        "1d_close_time",
        "daily_confirmation",
        "four_hour_confirmation",
        "4h_data_age_hours",
        "1d_data_age_hours",
    ]

    missing = [
        column for column in required_columns
        if column not in result.columns
    ]

    if missing:
        raise AssertionError(f"Missing output columns: {missing}")

    allowed_signals = {"LONG", "SHORT", "WAIT"}

    invalid_signals = set(result["signal"].unique()) - allowed_signals

    if invalid_signals:
        raise AssertionError(f"Invalid signals found: {invalid_signals}")

    if (result["4h_close_time"] > result["signal_time"]).any():
        raise AssertionError("Future 4H data detected.")

    if (result["1d_close_time"] > result["signal_time"]).any():
        raise AssertionError("Future 1D data detected.")

    if (result["4h_data_age_hours"] < 0).any():
        raise AssertionError("Negative 4H data age detected.")

    if (result["1d_data_age_hours"] < 0).any():
        raise AssertionError("Negative 1D data age detected.")

    if result["timestamp"].duplicated().any():
        raise AssertionError("Duplicate hourly timestamps found.")

    signal_counts = result["signal"].value_counts()

    print("\nSIGNAL SUMMARY")
    print("-" * 65)

    for signal in ["LONG", "SHORT", "WAIT"]:
        print(f"{signal}: {signal_counts.get(signal, 0)}")

    latest = result.iloc[-1]

    print("\nLATEST ENGINE OUTPUT")
    print("-" * 65)
    print(f"Signal: {latest['signal']}")
    print(f"Signal Type: {latest['signal_type']}")
    print(f"Daily Confirmation: {latest['daily_confirmation']}")
    print(f"4H Confirmation: {latest['four_hour_confirmation']}")
    print(f"Signal Time: {latest['signal_time']}")
    print(f"4H Data Age: {latest['4h_data_age_hours']:.2f} hours")
    print(f"1D Data Age: {latest['1d_data_age_hours']:.2f} hours")

    print("\nAligned Rows:", len(result))
    print("Future Data Matches: 0")
    print("Duplicate Hourly Timestamps: 0")

    print("\nSTATUS: PASS")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
