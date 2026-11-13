
import pandas as pd

from Src.Engines.Trendmomentum.data_loader import load_all_timeframes


TIMEFRAME_MS = {
    "1H": 3_600_000,
    "4H": 14_400_000,
    "1D": 86_400_000,
}


def format_timestamp(timestamp_ms):
    return pd.to_datetime(
        timestamp_ms,
        unit="ms",
        utc=True
    ).strftime("%Y-%m-%d %H:%M UTC")


def main():
    print("MULTI-TIMEFRAME ALIGNMENT TEST")
    print("================================")

    datasets = load_all_timeframes()

    hourly = datasets["1H"].copy()
    four_hour = datasets["4H"].copy()
    daily = datasets["1D"].copy()

    hourly["signal_close_ms"] = (
        hourly["timestamp"] + TIMEFRAME_MS["1H"]
    )

    four_hour["closed_at_ms"] = (
        four_hour["timestamp"] + TIMEFRAME_MS["4H"]
    )

    daily["closed_at_ms"] = (
        daily["timestamp"] + TIMEFRAME_MS["1D"]
    )

    four_hour = four_hour[
        ["timestamp", "closed_at_ms"]
    ].rename(
        columns={
            "timestamp": "4H_timestamp",
            "closed_at_ms": "4H_closed_at_ms"
        }
    )

    daily = daily[
        ["timestamp", "closed_at_ms"]
    ].rename(
        columns={
            "timestamp": "1D_timestamp",
            "closed_at_ms": "1D_closed_at_ms"
        }
    )

    aligned = pd.merge_asof(
        hourly.sort_values("signal_close_ms"),
        four_hour.sort_values("4H_closed_at_ms"),
        left_on="signal_close_ms",
        right_on="4H_closed_at_ms",
        direction="backward"
    )

    aligned = pd.merge_asof(
        aligned.sort_values("signal_close_ms"),
        daily.sort_values("1D_closed_at_ms"),
        left_on="signal_close_ms",
        right_on="1D_closed_at_ms",
        direction="backward"
    )

    all_passed = True

    missing_4h = aligned["4H_timestamp"].isna().sum()
    missing_1d = aligned["1D_timestamp"].isna().sum()

    future_4h = (
        aligned["4H_closed_at_ms"]
        > aligned["signal_close_ms"]
    ).sum()

    future_1d = (
        aligned["1D_closed_at_ms"]
        > aligned["signal_close_ms"]
    ).sum()

    print("\nALIGNMENT SUMMARY")
    print("--------------------------------")
    print("1H Candles:", len(aligned))
    print("Missing 4H Mapping:", missing_4h)
    print("Missing 1D Mapping:", missing_1d)
    print("Future 4H Data Matches:", future_4h)
    print("Future 1D Data Matches:", future_1d)

    if missing_4h > 0 or missing_1d > 0:
        all_passed = False

    if future_4h > 0 or future_1d > 0:
        all_passed = False

    latest = aligned.iloc[-1]

    print("\nLATEST VALID ALIGNMENT")
    print("--------------------------------")
    print(
        "1H Candle Open:",
        format_timestamp(latest["timestamp"])
    )
    print(
        "1H Candle Close:",
        format_timestamp(latest["signal_close_ms"])
    )
    print(
        "Matched 4H Candle Open:",
        format_timestamp(latest["4H_timestamp"])
    )
    print(
        "Matched 4H Candle Close:",
        format_timestamp(latest["4H_closed_at_ms"])
    )
    print(
        "Matched 1D Candle Open:",
        format_timestamp(latest["1D_timestamp"])
    )
    print(
        "Matched 1D Candle Close:",
        format_timestamp(latest["1D_closed_at_ms"])
    )

    age_4h = (
        latest["signal_close_ms"]
        - latest["4H_closed_at_ms"]
    ) / 3_600_000

    age_1d = (
        latest["signal_close_ms"]
        - latest["1D_closed_at_ms"]
    ) / 3_600_000

    print("\nDATA AGE AT SIGNAL TIME")
    print("--------------------------------")
    print(f"4H Data Age: {age_4h:.2f} hours")
    print(f"1D Data Age: {age_1d:.2f} hours")

    print("\nFINAL ALIGNMENT TEST")
    print("====================")
    print("RESULT:", "PASS" if all_passed else "FAIL")

    if not all_passed:
        raise RuntimeError(
            "Multi-timeframe alignment test failed."
        )


if __name__ == "__main__":
    main()
