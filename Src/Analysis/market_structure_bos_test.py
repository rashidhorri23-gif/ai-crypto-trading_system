
import sys
import numpy as np
import pandas as pd

from market_structure import detect_market_structure


LEFT_BARS = 2
RIGHT_BARS = 2


def build_data():
    count = 16

    timestamps = (
        1794000000000
        + np.arange(count) * 5 * 60 * 1000
    )

    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": np.full(count, 100.0),
        "high": np.full(count, 101.0),
        "low": np.full(count, 99.0),
        "close": np.full(count, 100.0),
        "volume": np.full(count, 10.0)
    })

    return data


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run_structure(data):
    return detect_market_structure(
        data,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS
    )


def test_bullish_bos():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "close"] = 111.0
    data.at[7, "high"] = 112.0

    result = run_structure(data)

    check(
        result.at[5, "swing_high_confirmed"],
        "Swing High was not confirmed."
    )

    check(
        result.at[7, "structure_event"] == "BOS_BULLISH",
        "Bullish BOS was not detected."
    )

    check(
        result.at[7, "broken_level"] == 110.0,
        "Incorrect bullish broken level."
    )

    check(
        result.at[7, "structure_trend"] == 1,
        "Incorrect bullish trend."
    )

    print("BULLISH BOS: PASS")


def test_bearish_bos():
    data = build_data()

    data.at[3, "low"] = 90.0

    data.at[7, "close"] = 89.0
    data.at[7, "low"] = 88.0

    result = run_structure(data)

    check(
        result.at[5, "swing_low_confirmed"],
        "Swing Low was not confirmed."
    )

    check(
        result.at[7, "structure_event"] == "BOS_BEARISH",
        "Bearish BOS was not detected."
    )

    check(
        result.at[7, "broken_level"] == 90.0,
        "Incorrect bearish broken level."
    )

    check(
        result.at[7, "structure_trend"] == -1,
        "Incorrect bearish trend."
    )

    print("BEARISH BOS: PASS")


def test_bullish_choch():
    data = build_data()

    data.at[3, "low"] = 90.0

    data.at[7, "close"] = 89.0
    data.at[7, "low"] = 88.0

    data.at[9, "high"] = 110.0

    data.at[13, "close"] = 111.0
    data.at[13, "high"] = 112.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "BOS_BEARISH",
        "Initial bearish BOS is missing."
    )

    check(
        result.at[13, "structure_event"] == "CHOCH_BULLISH",
        "Bullish CHOCH was not detected."
    )

    check(
        result.at[13, "structure_trend"] == 1,
        "Trend did not change to bullish."
    )

    print("BULLISH CHOCH: PASS")


def test_bearish_choch():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "close"] = 111.0
    data.at[7, "high"] = 112.0

    data.at[9, "low"] = 90.0

    data.at[13, "close"] = 89.0
    data.at[13, "low"] = 88.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "BOS_BULLISH",
        "Initial bullish BOS is missing."
    )

    check(
        result.at[13, "structure_event"] == "CHOCH_BEARISH",
        "Bearish CHOCH was not detected."
    )

    check(
        result.at[13, "structure_trend"] == -1,
        "Trend did not change to bearish."
    )

    print("BEARISH CHOCH: PASS")


def test_wick_does_not_break_structure():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "high"] = 112.0
    data.at[7, "close"] = 109.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "NONE",
        "Wick-only break was incorrectly detected."
    )

    check(
        result.at[7, "structure_trend"] == 0,
        "Trend changed without a candle close break."
    )

    print("WICK BREAK FILTER: PASS")


def test_single_break_per_level():
    data = build_data()

    data.at[3, "high"] = 110.0

    for index in range(7, 12):
        data.at[index, "close"] = 111.0
        data.at[index, "high"] = 111.5
        data.at[index, "low"] = 110.5

    result = run_structure(data)

    events = result[
        result["structure_event"] != "NONE"
    ]

    check(
        len(events) == 1,
        "The same broken level generated multiple events."
    )

    check(
        events.iloc[0]["structure_event"] == "BOS_BULLISH",
        "Unexpected structure event."
    )

    print("SINGLE BREAK PER LEVEL: PASS")


def test_event_timestamps():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "close"] = 111.0
    data.at[7, "high"] = 112.0

    result = run_structure(data)

    check(
        result.at[7, "broken_level_timestamp"]
        == result.at[3, "timestamp"],
        "Incorrect broken level timestamp."
    )

    check(
        result.at[7, "structure_event_timestamp"]
        == result.at[7, "timestamp"],
        "Incorrect event timestamp."
    )

    check(
        not (
            result.loc[:6, "structure_event"] != "NONE"
        ).any(),
        "Structure event was recorded before confirmation."
    )

    print("EVENT TIMESTAMP VALIDATION: PASS")


def test_output_integrity():
    data = build_data()
    result = run_structure(data)

    check(
        len(result) == len(data),
        "Output row count changed."
    )

    check(
        result["timestamp"].is_monotonic_increasing,
        "Timestamps are not ordered."
    )

    check(
        result["structure_event"].notna().all(),
        "Missing structure event values."
    )

    check(
        result["structure_trend"].isin([-1, 0, 1]).all(),
        "Invalid structure trend values."
    )

    print("OUTPUT INTEGRITY: PASS")


def main():
    print("BOS AND CHOCH UNIT TEST")
    print("=======================")
    print("Data Source: Synthetic Test Data")
    print("Left Bars:", LEFT_BARS)
    print("Right Bars:", RIGHT_BARS)
    print("Break Rule: Candle Close")
    print("Archive Modification: DISABLED")

    tests = [
        test_bullish_bos,
        test_bearish_bos,
        test_bullish_choch,
        test_bearish_choch,
        test_wick_does_not_break_structure,
        test_single_break_per_level,
        test_event_timestamps,
        test_output_integrity
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as error:
            failed += 1
            print(f"{test.__name__}: FAIL")
            print(f"ERROR: {error}")

    print("\nFINAL BOS AND CHOCH REPORT")
    print("==========================")
    print("Passed:", passed)
    print("Failed:", failed)

    print(
        "OVERALL STATUS:",
        "PASS" if failed == 0 else "FAIL"
    )

    print("ARCHIVE MODIFIED: NO")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
