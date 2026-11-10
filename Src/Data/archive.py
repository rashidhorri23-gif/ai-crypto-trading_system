
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

CSV_FIELDS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "market",
    "symbol",
    "timeframe",
    "source"
]


def get_latest_timestamp(filename):
    archive_path = ARCHIVE_DIR / filename

    if not archive_path.exists():
        return None

    latest_timestamp = None

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            timestamp = int(row["timestamp"])

            if latest_timestamp is None:
                latest_timestamp = timestamp
            else:
                latest_timestamp = max(
                    latest_timestamp,
                    timestamp
                )

    return latest_timestamp


def save_candles(candles, filename):
    if not candles:
        raise ValueError("No candles provided for archiving.")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = ARCHIVE_DIR / filename
    existing = {}

    if archive_path.exists():
        with archive_path.open(
            "r",
            newline="",
            encoding="utf-8"
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                timestamp = int(row["timestamp"])
                existing[timestamp] = row

    previous_count = len(existing)

    for candle in candles:
        timestamp = int(candle["timestamp"])

        existing[timestamp] = {
            field: candle[field]
            for field in CSV_FIELDS
        }

    sorted_rows = sorted(
        existing.values(),
        key=lambda row: int(row["timestamp"])
    )

    with archive_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()
        writer.writerows(sorted_rows)

    return {
        "file": str(archive_path.relative_to(PROJECT_ROOT)),
        "previous_candles": previous_count,
        "total_candles": len(sorted_rows),
        "new_candles": len(sorted_rows) - previous_count
    }
