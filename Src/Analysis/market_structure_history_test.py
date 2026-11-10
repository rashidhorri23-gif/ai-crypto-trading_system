
from pathlib import Path
import sys
import math
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_structure import detect_market_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]
PREFIX_RATIOS = [0.50, 0.70, 0.85]

LEFT_BARS = 3
RIGHT_BARS = 3

COMPARE_COLUMNS = [
    "swing_high_confirmed",
    "swing_low_confirmed",
    "swing_high_price",
    "swing_low_price",
    "swing_high_pivot_timestamp",
    "swing_low_pivot_timestamp",
    "swing_high_confirmation_timestamp",
    "swing_low_confirmation_timestamp",
    "structure_event",
    "structure_direction",
    "structure_trend",
    "broken_level",
    "broken_level_timestamp",
    "structure_event_timestamp",
]


def values_equal(first, second):
    first_missing = pd.isna(first)
    second_missing = pd.isna(second)

    if first_missing and second_missing:
        return True

    if first_missing or second_missing:
        return False

    if isinstance(first, (bool,)) or isinstance(second, (bool,)):
        return first == second

    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(
            float(first),
            float(second),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

    return str(first) == str(second)


def create_timestamp_map(result):
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in analysis output")

    return {
        int(row["timestamp"]): row
        for _, row in result.iterrows()
    }


def compare_prefix(prefix_result, full_map):
    differences = []

    for _, prefix_row in prefix_result.iterrows():
        timestamp = int(prefix_row["timestamp"])

        if timestamp not in full_map:
            differences.append(
                f"Timestamp {timestamp} missing from full result"
            )
            continue

        full_row = full_map[timestamp]

        for column in COMPARE_COLUMNS:
            first = prefix_row[column]
            second = full_row[column]

            if not values_equal(first, second):
                differences.append(
                    f"Timestamp {timestamp}, column {column}: "
                    f"prefix={first}, full={second}"
                )

    return differences


def audit_timeframe(timeframe):
    filename = f"BTC_USDT_SPOT_{timeframe}.csv"
    file_path = ARCHIVE_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Archive not found: {filename}")

    candles = pd.read_csv(file_path)

    if len(candles) < 50:
        raise ValueError("Not enough candles for prefix testing")

    full_result = detect_market_structure(
        candles,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
    )

    missing = [
        column
        for column in ["timestamp"] + COMPARE_COLUMNS
        if column not in full_result.columns
    ]

    if missing:
        raise ValueError(f"Missing output columns: {missing}")

    full_map = create_timestamp_map(full_result)

    total_comparisons = 0
    total_differences = 0
    errors = []

    print("")
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(candles)}")

    for ratio in PREFIX_RATIOS:
        prefix_size = int(len(candles) * ratio)

        prefix_candles = candles.iloc[:prefix_size].copy()

        prefix_result = detect_market_structure(
            prefix_candles,
            left_bars=LEFT_BARS,
            right_bars=RIGHT_BARS,
        )

        differences = compare_prefix(
            prefix_result,
            full_map,
        )

        total_comparisons += len(prefix_result)
        total_differences += len(differences)

        status = "PASS" if not differences else "FAIL"

        print(
            f"Prefix {int(ratio * 100)}%: "
            f"Candles={prefix_size}, "
            f"Compared={len(prefix_result)}, "
            f"Differences={len(differences)}, "
            f"Status={status}"
        )

        for difference in differences[:5]:
            errors.append(
                f"{timeframe} {int(ratio * 100)}%: {difference}"
            )

    status = "PASS" if total_differences == 0 else "FAIL"

    print(f"Total Compared Rows: {total_comparisons}")
    print(f"Total Differences: {total_differences}")
    print(f"TIMEFRAME STATUS: {status}")

    return {
        "timeframe": timeframe,
        "candles": len(candles),
        "comparisons": total_comparisons,
        "differences": total_differences,
        "status": status,
        "errors": errors,
    }


def main():
    print("PREFIX INVARIANCE TEST")
    print("======================")
    print(f"Left Bars: {LEFT_BARS}")
    print(f"Right Bars: {RIGHT_BARS}")
    print(f"Prefix Ratios: {PREFIX_RATIOS}")
    print("Market: BTC-USDT SPOT")
    print("Data Source: Local OKX Archive")
    print("Archive Modification: DISABLED")

    reports = []

    for timeframe in TIMEFRAMES:
        try:
            report = audit_timeframe(timeframe)
            reports.append(report)

        except Exception as error:
            print("")
            print(f"TIMEFRAME: {timeframe}")
            print(f"FATAL ERROR: {error}")

            reports.append({
                "timeframe": timeframe,
                "candles": 0,
                "comparisons": 0,
                "differences": 1,
                "status": "FAIL",
                "errors": [str(error)],
            })

    print("")
    print("FINAL PREFIX INVARIANCE REPORT")
    print("==============================")

    for report in reports:
        print(
            f"{report['timeframe']}: "
            f"Candles={report['candles']}, "
            f"Compared={report['comparisons']}, "
            f"Differences={report['differences']}, "
            f"Status={report['status']}"
        )

        for error in report["errors"][:3]:
            print(f"ERROR: {error}")

    passed = sum(
        1 for report in reports
        if report["status"] == "PASS"
    )

    failed = len(reports) - passed

    print("")
    print(f"Passed Timeframes: {passed}")
    print(f"Failed Timeframes: {failed}")
    print(f"Total Timeframes: {len(reports)}")

    overall = "PASS" if failed == 0 else "FAIL"

    print(f"OVERALL PREFIX INVARIANCE: {overall}")
    print("ARCHIVE MODIFICATION: NO")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
