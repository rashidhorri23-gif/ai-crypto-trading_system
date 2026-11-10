
from pathlib import Path
import sys
import pandas as pd
import math

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
    "direction",
    "broken_level",
    "broken_level_timestamp",
    "structure_event_timestamp",
    "swing_high",
    "swing_low",
    "swing_high_price",
    "swing_low_price",
    "swing_high_timestamp",
    "swing_low_timestamp",
    "swing_high_confirmation_timestamp",
    "swing_low_confirmation_timestamp",
]


def build_swing_map(result, swing_type):
    if swing_type == "HIGH":
        flag_col = "swing_high"
        price_col = "swing_high_price"
        pivot_col = "swing_high_timestamp"
        confirm_col = "swing_high_confirmation_timestamp"
    else:
        flag_col = "swing_low"
        price_col = "swing_low_price"
        pivot_col = "swing_low_timestamp"
        confirm_col = "swing_low_confirmation_timestamp"

    mask = result[flag_col].fillna(False).astype(bool)
    swings = result.loc[mask, [price_col, pivot_col, confirm_col]]

    swing_map = {}

    for _, row in swings.iterrows():
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

    missing = [col for col in REQUIRED_COLUMNS if col not in result.columns]

    if missing:
        raise ValueError(
            f"{timeframe}: Missing output columns: {missing}"
        )

    high_map = build_swing_map(result, "HIGH")
    low_map = build_swing_map(result, "LOW")

    events = result[
        result["structure_event"].notna()
        & result["structure_event"].astype(str).str.strip().ne("")
    ].copy()

    counts = {
        "BULLISH_BOS": 0,
        "BEARISH_BOS": 0,
        "BULLISH_CHOCH": 0,
        "BEARISH_CHOCH": 0,
    }

    errors = []

    for row_number, row in events.iterrows():
        event_name = str(row["structure_event"]).strip().upper()
        direction = str(row["direction"]).strip().upper()

        timestamp = int(row["timestamp"])
        event_timestamp = int(row["structure_event_timestamp"])
        pivot_timestamp = int(row["broken_level_timestamp"])
        broken_level = float(row["broken_level"])

        if event_timestamp != timestamp:
            errors.append(
                f"Row {row_number}: Event timestamp does not match candle timestamp"
            )

        if event_name not in ("BOS", "CHOCH"):
            errors.append(
                f"Row {row_number}: Unknown event type: {event_name}"
            )
            continue

        if direction not in ("BULLISH", "BEARISH"):
            errors.append(
                f"Row {row_number}: Unknown direction: {direction}"
            )
            continue

        key = f"{direction}_{event_name}"
        counts[key] += 1

        swing_map = high_map if direction == "BULLISH" else low_map

        if pivot_timestamp not in swing_map:
            errors.append(
                f"Row {row_number}: Broken swing not found: {pivot_timestamp}"
            )
            continue

        swing = swing_map[pivot_timestamp]
        confirmation_timestamp = swing["confirmation"]
        swing_price = swing["price"]

        if confirmation_timestamp > event_timestamp:
            errors.append(
                f"Row {row_number}: LOOKAHEAD ERROR - event occurred before swing confirmation"
            )

        if pivot_timestamp > confirmation_timestamp:
            errors.append(
                f"Row {row_number}: Pivot timestamp is after confirmation timestamp"
            )

        if not math.isclose(
            broken_level,
            swing_price,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            errors.append(
                f"Row {row_number}: Broken level does not match swing price"
            )

    total_events = len(events)

    print("")
    print(f"TIMEFRAME: {timeframe}")
    print(f"Archive Candles: {len(candles)}")
    print(f"Detected Events: {total_events}")
    print(f"Bullish BOS: {counts['BULLISH_BOS']}")
    print(f"Bearish BOS: {counts['BEARISH_BOS']}")
    print(f"Bullish CHOCH: {counts['BULLISH_CHOCH']}")
    print(f"Bearish CHOCH: {counts['BEARISH_CHOCH']}")
    print(f"Swing Highs: {len(high_map)}")
    print(f"Swing Lows: {len(low_map)}")
    print(f"Audit Errors: {len(errors)}")

    for error in errors[:10]:
        print(f"ERROR: {error}")

    if len(errors) > 10:
        print(f"Additional errors not displayed: {len(errors) - 10}")

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

    passed = sum(1 for report in reports if report["status"] == "PASS")
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
