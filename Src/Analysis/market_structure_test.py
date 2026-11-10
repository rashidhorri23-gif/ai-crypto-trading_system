import sys
import numpy as np
import pandas as pd

from market_structure import detect_swings


LEFT_BARS = 2
RIGHT_BARS = 2


def build_test_data():
    count = 24

    timestamps = (
        1794000000000
        + np.arange(count) * 5 * 60 * 1000
    )

    open_prices = np.full(count, 100.0)
    close_prices = np.full(count, 100.0)
    high_prices = np.full(count, 105.0)
    low_prices = np.full(count, 95.0)
    volume = np.full(count, 10.0)

    high_prices[5] = 120.0
    low_prices[10] = 80.0

    high_prices[15] = 125.0
    high_prices[16] = 125.0

    low_prices[18] = 75.0
    low_prices[19] = 75.0

    high_prices[23] = 130.0

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume
    })


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_swing_high(result):
    rows = result.index[
        result["swing_high_confirmed"]
    ].tolist()

    check(
        7 in rows,
        "Expected Swing High confirmation at index 7."
    )

    check(
        result.at[7, "swing_high_price"] == 120.0,
        "Incorrect Swing High price."
    )

    check(
        result.at[7, "swing_high_pivot_timestamp"]
        == result.at[5, "timestamp"],
        "Incorrect Swing High pivot timestamp."
    )

    check(
        result.at[7, "swing_high_confirmation_timestamp"]
        == result.at[7, "timestamp"],
        "Incorrect Swing High confirmation timestamp."
    )

    check(
        not result["swing_high_confirmed"].iloc[:7].any(),
        "Swing High was confirmed too early."
    )

    check(
        15 not in rows and 16 not in rows,
        "Duplicate high values must not create Swing High."
    )

    check(
        23 not in rows,
        "Unconfirmed final high must not create Swing High."
    )

    print("SWING HIGH DETECTION: PASS")


def test_swing_low(result):
    rows = result.index[
        result["swing_low_confirmed"]
    ].tolist()

    check(
        12 in rows,
        "Expected Swing Low confirmation at index 12."
    )

    check(
        result.at[12, "swing_low_price"] == 80.0,
        "Incorrect Swing Low price."
    )

    check(
        result.at[12, "swing_low_pivot_timestamp"]
        == result.at[10, "timestamp"],
        "Incorrect Swing Low pivot timestamp."
    )

    check(
        result.at[12, "swing_low_confirmation_timestamp"]
        == result.at[12, "timestamp"],
        "Incorrect Swing Low confirmation timestamp."
    )

    check(
        not result["swing_low_confirmed"].iloc[:12].any(),
        "Swing Low was confirmed too early."
    )

    check(
        18 not in rows and 19 not in rows,
        "Duplicate low values must not create Swing Low."
    )

    print("SWING LOW DETECTION: PASS")


def test_confirmation_delay(result):
    high_confirmation = result.index[
        result["swing_high_confirmed"]
    ].tolist()

    low_confirmation = result.index[
        result["swing_low_confirmed"]
    ].tolist()

    for index in high_confirmation:
        pivot_timestamp = result.at[
            index,
            "swing_high_pivot_timestamp"
        ]

        confirmation_timestamp = result.at[
            index,
            "swing_high_confirmation_timestamp"
        ]

        pivot_index = result.index[
            result["timestamp"] == pivot_timestamp
        ][0]

        check(
            index - pivot_index == RIGHT_BARS,
            "Incorrect Swing High confirmation delay."
        )

        check(
            confirmation_timestamp == result.at[index, "timestamp"],
            "Invalid Swing High confirmation time."
        )

    for index in low_confirmation:
        pivot_timestamp = result.at[
            index,
            "swing_low_pivot_timestamp"
        ]

        confirmation_timestamp = result.at[
            index,
            "swing_low_confirmation_timestamp"
        ]

        pivot_index = result.index[
            result["timestamp"] == pivot_timestamp
        ][0]

        check(
            index - pivot_index == RIGHT_BARS,
            "Incorrect Swing Low confirmation delay."
        )

        check(
            confirmation_timestamp == result.at[index, "timestamp"],
            "Invalid Swing Low confirmation time."
        )

    print("CONFIRMATION DELAY: PASS")


def test_output_integrity(result):
    check(
        len(result) == 24,
        "Output row count changed."
    )

    check(
        result["timestamp"].is_monotonic_increasing,
        "Output timestamps are not ordered."
    )

    check(
        not result[
            ["swing_high_confirmed", "swing_low_confirmed"]
        ].isnull().any().any(),
        "Swing confirmation flags contain null values."
    )

    print("OUTPUT INTEGRITY: PASS")


def main():
    print("MARKET STRUCTURE UNIT TEST")
    print("==========================")
    print(f"Left Bars: {LEFT_BARS}")
    print(f"Right Bars: {RIGHT_BARS}")
    print("Data Source: Synthetic Test Data")
    print("Archive Modification: DISABLED")

    all_passed = True

    try:
        data = build_test_data()

        result = detect_swings(
            data,
            left_bars=LEFT_BARS,
            right_bars=RIGHT_BARS
        )

        test_swing_high(result)
        test_swing_low(result)
        test_confirmation_delay(result)
        test_output_integrity(result)

    except Exception as error:
        all_passed = False
        print(f"TEST ERROR: {error}")

    print("\nFINAL MARKET STRUCTURE REPORT")
    print("=============================")

    print(
        "OVERALL MARKET STRUCTURE STATUS: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print("ARCHIVE MODIFIED: NO")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
