
from pathlib import Path
import sys
import math
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from market_structure import detect_market_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]

LEFT_BARS = 3
RIGHT_BARS = 3

REQUIRED_COLUMNS = [
    "timestamp",
    "structure_event",
    "structure_direction",
    "structure_trend",
    "broken_level",
    "broken_level_timestamp",
    "structure_event_timestamp",
    "swing_high_confirmed",
    "swing_low_confirmed",
    "swing_high_price",
    "swing_low_price",
    "swing_high_pivot_timestamp",
    "swing_low_pivot_timestamp",
    "swing_high_confirmation_timestamp",
    "swing_low_confirmation_timestamp",
]

VALID_EVENTS = {
    "NONE",
    "BOS_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BULLISH",
    "CHOCH_BEARISH",
}


def build_swing_map(result, swing_type):
    if swing_type == "HIGH":
        flag_col = "swing_high_confirmed"
        price_col = "swing_high_price"
        pivot_col = "swing_high_pivot_timestamp"
        confirm_col = "swing_high_confirmation_timestamp"
    else:
        flag_col = "swing_low_confirmed"
        price_col = "swing_low_price"
        pivot_col = "swing_low_pivot_timestamp"
        confirm_col = "swing_low_confirmation_timestamp"

    mask = result[flag_col].fillna(False).astype(bool)
    swings = result.loc[
        mask,
        [price_col, pivot_col, confirm_col]
    ]

    swing_map = {}

    for _, row in swings.iterrows():
        if pd.isna(row[pivot_col]) or pd.isna(row[confirm_col]):
            continue

        pivot_time = int(row[pivot_col])
        confirm_time = int(row[confirm_col])
        price = float(row[price_col])

        swing_map[pivot_time] = {
            "confirmation": confirm_time,
            "price": price,
        }

    return swing_map


def audit_timeframe(timeframe):
    filename = f"BTC_USDT_SPOT_{timeframe}.csv"
    file_path = ARCHIVE_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Archive not found: {filename}")

    candles = pd.read_csv(file_path)

    if candles.empty:
        raise ValueError(f"Archive is empty: {filename}")

    result = detect_market_structure(
        candles,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
    )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in result.columns
    ]

    if missing:
        raise ValueError(
            f"{timeframe}: Missing output columns: {missing}"
        )

    high_map = build_swing_map(result, "HIGH")
    low_map = build_swing_map(result, "LOW")

    events = result[
        result["structure_event"].isin([
            "BOS_BULLISH",
            "BOS_BEARISH",
            "CHOCH_BULLISH",
            "CHOCH_BEARISH",
        ])
    ].copy()

    counts = {
        "BOS_BULLISH": 0,
        "BOS_BEARISH": 0,
        "CHOCH_BULLISH": 0,
        "CHOCH_BEARISH": 0,
    }

    errors = []
    checked_events = 0

    for row_number, row in events.iterrows():
        event_name = str(row["structure_event"])
        direction = int(row["structure_direction"])
        trend = int(row["structure_trend"])

        timestamp = int(row["timestamp"])

        if pd.isna(row["structure_event_timestamp"]):
            errors.append(
                f"Row {row_number}: Missing event timestamp"
            )
            continue

        if pd.isna(row["broken_level_timestamp"]):
            errors.append(
                f"Row {row_number}: Missing broken level timestamp"
            )
            continue

        if pd.isna(row["broken_level"]):
            errors.append(
                f"Row {row_number}: Missing broken level price"
            )
            continue

        event_timestamp = int(row["structure_event_timestamp"])
        pivot_timestamp = int(row["broken_level_timestamp"])
        broken_level = float(row["broken_level"])

        counts[event_name] += 1
        checked_events += 1

        if event_timestamp != timestamp:
            errors.append(
                f"Row {row_number}: Event timestamp mismatch"
            )

        expected_direction = (
            1 if event_name.endswith("BULLISH") else -1
        )

        if direction != expected_direction:
            errors.append(
                f"Row {row_number}: Direction mismatch"
            )

        if trend != expected_direction:
            errors.append(
                f"Row {row_number}: Trend mismatch"
            )

        swing_map = (
            high_map if expected_direction == 1 else low_map
        )

        if pivot_timestamp not in swing_map:
            errors.append(
                f"Row {row_number}: Broken swing not found"
            )
            continue

        swing = swing_map[pivot_timestamp]

        confirmation_timestamp = swing["confirmation"]
        swing_price = swing["price"]

        if confirmation_timestamp > event_timestamp:
            errors.append(
                f"Row {row_number}: LOOKAHEAD ERROR"
            )

        if pivot_timestamp > confirmation_timestamp:
            errors.append(
                f"Row {row_number}: Invalid swing timestamps"
            )

        if not math.isclose(
            broken_level,
            swing_price,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            errors.append(
                f"Row {row_number}: Broken level price mismatch"
            )

        if event_timestamp <= confirmation_timestamp:
            errors.append(
                f"Row {row_number}: Event must occur after swing confirmation"
            )

    total_events = len(events)

    print("")
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(candles)}")
    print(f"Swing Highs: {len(high_map)}")
    print(f"Swing Lows: {len(low_map)}")
    print(f"Total Events: {total_events}")
    print(f"Checked Events: {checked_events}")

    print(f"Bullish BOS: {counts['BOS_BULLISH']}")
    print(f"Bearish BOS: {counts['BOS_BEARISH']}")
    print(f"Bullish CHOCH: {counts['CHOCH_BULLISH']}")
    print(f"Bearish CHOCH: {counts['CHOCH_BEARISH']}")

    print(f"Audit Errors: {len(errors)}")

    for error in errors[:10]:
        print(f"ERROR: {error}")

    if len(errors) > 10:
        print(
            f"Additional errors not displayed: {len(errors) - 10}"
        )

    status = "PASS" if not errors else "FAIL"

    print(f"TIMEFRAME STATUS: {status}")

    return {
        "timeframe": timeframe,
        "candles": len(candles),
        "events": total_events,
        "errors": len(errors),
        "status": status,
    }


def main():
    print("REAL MARKET STRUCTURE HISTORY AUDIT")
    print("====================================")
    print(f"Left Bars: {LEFT_BARS}")
    print(f"Right Bars: {RIGHT_BARS}")
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
                "events": 0,
                "errors": 1,
                "status": "FAIL",
            })

    print("")
    print("FINAL HISTORICAL AUDIT REPORT")
    print("=============================")

    for report in reports:
        print(
            f"{report['timeframe']}: "
            f"Candles={report['candles']}, "
            f"Events={report['events']}, "
            f"Errors={report['errors']}, "
            f"Status={report['status']}"
        )

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

    print(f"OVERALL HISTORICAL AUDIT: {overall}")
    print("ARCHIVE MODIFICATION: NO")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
