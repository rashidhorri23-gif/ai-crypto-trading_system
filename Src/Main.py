
import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings
from Src.Data.okx_client import fetch_new_candles
from Src.Data.archive import save_candles, get_latest_timestamp


FILENAME = "BTC_USDT_SPOT_5m.csv"
INTERVAL_MS = 5 * 60 * 1000


def audit_archive(filename):
    archive_path = (
        PROJECT_ROOT / "Data" / "Archive" / filename
    )

    if not archive_path.exists():
        raise RuntimeError("Archive file not found.")

    timestamps = []
    invalid_rows = 0
    misaligned_timestamps = 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                timestamp = int(row["timestamp"])

                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
                volume = float(row["volume"])

                if (
                    min(
                        open_price,
                        high_price,
                        low_price,
                        close_price
                    ) <= 0
                    or volume < 0
                    or high_price < max(
                        open_price,
                        close_price,
                        low_price
                    )
                    or low_price > min(
                        open_price,
                        close_price,
                        high_price
                    )
                ):
                    invalid_rows += 1

                timestamps.append(timestamp)

                if timestamp % INTERVAL_MS != 0:
                    misaligned_timestamps += 1

            except (ValueError, KeyError, TypeError):
                invalid_rows += 1

    if not timestamps:
        raise RuntimeError("Archive contains no valid timestamps.")

    sorted_timestamps = sorted(timestamps)

    duplicate_count = (
        len(sorted_timestamps) - len(set(sorted_timestamps))
    )

    unique_timestamps = sorted(set(sorted_timestamps))

    missing_candles = 0
    gap_count = 0

    for index in range(1, len(unique_timestamps)):
        difference = (
            unique_timestamps[index]
            - unique_timestamps[index - 1]
        )

        if difference > INTERVAL_MS:
            missing = (
                difference // INTERVAL_MS
            ) - 1

            missing_candles += missing
            gap_count += 1

    first_time = datetime.fromtimestamp(
        unique_timestamps[0] / 1000,
        tz=timezone.utc
    )

    last_time = datetime.fromtimestamp(
        unique_timestamps[-1] / 1000,
        tz=timezone.utc
    )

    expected_count = (
        (unique_timestamps[-1] - unique_timestamps[0])
        // INTERVAL_MS
    ) + 1

    continuous = (
        len(unique_timestamps) == expected_count
    )

    print("\nARCHIVE AUDIT REPORT")
    print("====================")
    print(f"Archive File: {filename}")
    print(f"Total Rows: {len(timestamps)}")
    print(f"Unique Candles: {len(unique_timestamps)}")
    print(f"Duplicate Rows: {duplicate_count}")
    print(f"Missing Candles: {missing_candles}")
    print(f"Gap Segments: {gap_count}")
    print(f"Invalid Rows: {invalid_rows}")
    print(f"Misaligned Timestamps: {misaligned_timestamps}")
    print(f"First Candle UTC: {first_time.isoformat()}")
    print(f"Last Candle UTC: {last_time.isoformat()}")
    print(f"Expected Continuous Count: {expected_count}")
    print(f"Continuous Timeline: {continuous}")

    if (
        duplicate_count == 0
        and missing_candles == 0
        and invalid_rows == 0
        and misaligned_timestamps == 0
    ):
        print("AUDIT STATUS: PASS")
    else:
        print("AUDIT STATUS: ISSUES DETECTED")

    return {
        "total_rows": len(timestamps),
        "unique_candles": len(unique_timestamps),
        "duplicates": duplicate_count,
        "missing_candles": missing_candles,
        "gap_segments": gap_count,
        "invalid_rows": invalid_rows,
        "misaligned_timestamps": misaligned_timestamps,
        "continuous": continuous
    }


def main():
    print("AI Crypto Trading System")
    print("========================")
    print(f"Provider: {settings.DATA_PROVIDER}")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Timeframe: 5m")
    print("Mode: INCREMENTAL ARCHIVE + AUDIT")
    print("Live Trading: DISABLED")

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError("Live trading must remain disabled.")

    last_timestamp = get_latest_timestamp(FILENAME)

    if last_timestamp is None:
        raise RuntimeError("Existing archive not found.")

    last_time = datetime.fromtimestamp(
        last_timestamp / 1000,
        tz=timezone.utc
    )

    print(f"\nLast Archived UTC: {last_time.isoformat()}")

    candles = fetch_new_candles(
        market="SPOT",
        timeframe="5m",
        last_timestamp=last_timestamp,
        page_size=100
    )

    print("\nIncremental Fetch Results")
    print("=========================")
    print(f"New Candles Retrieved: {len(candles)}")

    if candles:
        result = save_candles(candles, FILENAME)

        print(f"New Candles Saved: {result['new_candles']}")
        print(f"Total Archived Candles: {result['total_candles']}")

        newest_time = datetime.fromtimestamp(
            candles[-1]["timestamp"] / 1000,
            tz=timezone.utc
        )

        print(f"Newest Candle UTC: {newest_time.isoformat()}")

    else:
        print("No new completed candles found.")
        print("Archive remains unchanged.")

    audit_archive(FILENAME)

    print("\nINCREMENTAL UPDATE AND AUDIT COMPLETED")
    print("LIVE TRADING: DISABLED")


if __name__ == "__main__":
    main()
