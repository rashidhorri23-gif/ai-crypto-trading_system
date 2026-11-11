
import pandas as pd

from market_structure import detect_market_structure


LEFT_BARS = 2
RIGHT_BARS = 2


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def build_data():
    rows = []

    for index in range(24):
        rows.append({
            "timestamp": 1700000000000 + index * 300000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0
        })

    return pd.DataFrame(rows)


def run_structure(data):
    return detect_market_structure(
        data,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS
    )


def test_bullish_bos():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "open"] = 109.0
    data.at[7, "close"] = 111.0
    data.at[7, "high"] = 112.0
    data.at[7, "low"] = 108.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "BOS_BULLISH",
        "Bullish BOS was not detected."
    )

    print("BULLISH BOS: PASS")


def test_bearish_bos():
    data = build_data()

    data.at[3, "low"] = 90.0

    data.at[7, "open"] = 91.0
    data.at[7, "close"] = 89.0
    data.at[7, "high"] = 92.0
    data.at[7, "low"] = 88.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "BOS_BEARISH",
        "Bearish BOS was not detected."
    )

    print("BEARISH BOS: PASS")


def test_bullish_choch():
    data = build_data()

    data.at[3, "low"] = 90.0

    data.at[7, "open"] = 91.0
    data.at[7, "close"] = 89.0
    data.at[7, "high"] = 92.0
    data.at[7, "low"] = 88.0

    data.at[9, "high"] = 110.0

    data.at[13, "open"] = 109.0
    data.at[13, "close"] = 111.0
    data.at[13, "high"] = 112.0
    data.at[13, "low"] = 108.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "BOS_BEARISH",
        "Initial bearish BOS was not detected."
    )

    check(
        result.at[13, "structure_event"] == "CHOCH_BULLISH",
        "Bullish CHOCH was not detected."
    )

    print("BULLISH CHOCH: PASS")


def test_bearish_choch():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "open"] = 109.0
    data.at[7, "close"] = 111.0
    data.at[7, "high"] = 112.0
    data.at[7, "low"] = 108.0

    data.at[9, "low"] = 90.0

    data.at[13, "open"] = 91.0
    data.at[13, "close"] = 89.0
    data.at[13, "high"] = 92.0
    data.at[13, "low"] = 88.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "BOS_BULLISH",
        "Initial bullish BOS was not detected."
    )

    check(
        result.at[13, "structure_event"] == "CHOCH_BEARISH",
        "Bearish CHOCH was not detected."
    )

    print("BEARISH CHOCH: PASS")


def test_wick_break_filter():
    data = build_data()

    data.at[3, "high"] = 110.0

    data.at[7, "open"] = 100.0
    data.at[7, "close"] = 109.0
    data.at[7, "high"] = 111.0
    data.at[7, "low"] = 99.0

    result = run_structure(data)

    check(
        result.at[7, "structure_event"] == "NONE",
        "Wick break incorrectly generated a structure event."
    )

    print("WICK BREAK FILTER: PASS")


def test_single_break_per_level():
    data = build_data()

    data.at[3, "high"] = 110.0

    for index in range(7, 12):
        data.at[index, "open"] = 110.5
        data.at[index, "close"] = 111.0
        data.at[index, "high"] = 111.5
        data.at[index, "low"] = 109.5

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

    data.at[7, "open"] = 109.0
    data.at[7, "close"] = 111.0
    data.at[7, "high"] = 112.0
    data.at[7, "low"] = 108.0

    result = run_structure(data)

    check(
        int(result.at[7, "structure_event_timestamp"])
        == int(data.at[7, "timestamp"]),
        "Incorrect event timestamp."
    )

    check(
        int(result.at[7, "broken_level_timestamp"])
        == int(data.at[3, "timestamp"]),
        "Incorrect broken level timestamp."
    )

    print("EVENT TIMESTAMPS: PASS")


def test_output_integrity():
    data = build_data()
    data.at[3, "high"] = 110.0

    result = run_structure(data)

    required_columns = [
        "swing_high_confirmed",
        "swing_low_confirmed",
        "structure_event",
        "structure_direction",
        "structure_trend",
        "broken_level",
        "broken_level_timestamp",
        "structure_event_timestamp"
    ]

    for column in required_columns:
        check(
            column in result.columns,
            f"Missing output column: {column}"
        )

    allowed_events = {
        "NONE",
        "BOS_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BULLISH",
        "CHOCH_BEARISH"
    }

    check(
        set(result["structure_event"].unique()).issubset(
            allowed_events
        ),
        "Unexpected structure event value."
    )

    check(
        len(result) == len(data),
        "Output row count changed."
    )

    print("OUTPUT INTEGRITY: PASS")


def main():
    print()
    print("BOS AND CHOCH UNIT TEST")
    print("=======================")
    print(f"Left Bars: {LEFT_BARS}")
    print(f"Right Bars: {RIGHT_BARS}")
    print("Data Source: Synthetic Test Data")
    print("Archive Modification: DISABLED")
    print()

    test_functions = [
        test_bullish_bos,
        test_bearish_bos,
        test_bullish_choch,
        test_bearish_choch,
        test_wick_break_filter,
        test_single_break_per_level,
        test_event_timestamps,
        test_output_integrity
    ]

    passed = 0
    failed = 0

    for function in test_functions:
        try:
            function()
            passed += 1
        except Exception as error:
            failed += 1
            print(f"{function.__name__}: FAIL - {error}")

    print()
    print("FINAL BOS AND CHOCH REPORT")
    print("==========================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if passed == 8 and failed == 0:
        print("OVERALL BOS AND CHOCH STATUS: PASS")
    else:
        print("OVERALL BOS AND CHOCH STATUS: FAIL")
        raise SystemExit(1)

    print("ARCHIVE MODIFICATION: NO")


if __name__ == "__main__":
    main()
