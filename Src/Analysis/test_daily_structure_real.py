
import csv
from pathlib import Path

from Src.Analysis.daily_structure import analyze_daily_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Archive"
    / "BTC_USDT_SPOT_1D.csv"
)


def format_timestamp(timestamp):
    from datetime import datetime, timezone

    timestamp = int(float(timestamp))

    if timestamp > 100000000000:
        timestamp //= 1000

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_candles():
    if not ARCHIVE_FILE.exists():
        raise FileNotFoundError(
            f"Archive file not found: {ARCHIVE_FILE}"
        )

    candles = []

    with ARCHIVE_FILE.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            candles.append({
                "timestamp": int(float(row["timestamp"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

    return candles


def print_structure(title, result):
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)

    print("Trend:", result.get("trend"))
    print("Event Bias:", result.get("event_bias"))

    structure = result.get("structure", {})

    print("Structure Reason:", structure.get("reason"))
    print("Swing High Count:", structure.get("high_count"))
    print("Swing Low Count:", structure.get("low_count"))

    for key in ("previous_high", "latest_high",
                "previous_low", "latest_low"):
        point = structure.get(key)

        if point:
            print(
                key,
                "Price:", point["price"],
                "Time:", format_timestamp(point["timestamp"]),
                "Confirmed:",
                format_timestamp(point["confirmation_timestamp"])
            )

    events = result.get("events", [])

    print("\nStructure Events:", len(events))

    for event in events:
        print(
            event["event"],
            "| Level:", event.get("level", "N/A"),
            "| Close:", event.get("close", "N/A"),
            "| Time:", format_timestamp(event["timestamp"]),
            "| Confirmation:", event.get("confirmation", "N/A")
        )


def run_real_data_test():
    print("REAL OKX DAILY STRUCTURE TEST")
    print("=" * 55)

    candles = load_candles()

    print("Source:", ARCHIVE_FILE.name)
    print("Total candles:", len(candles))
    print("First candle:", format_timestamp(candles[0]["timestamp"]))
    print("Last candle:", format_timestamp(candles[-1]["timestamp"]))
    print("Latest close:", candles[-1]["close"])

    result = analyze_daily_structure(candles)

    print_structure(
        "INTERNAL STRUCTURE",
        result["internal_structure"]
    )

    print_structure(
        "EXTERNAL STRUCTURE",
        result["external_structure"]
    )

    print("\n" + "=" * 55)
    print("FINAL DAILY DECISION")
    print("=" * 55)

    print("Allowed Direction:", result["allowed_direction"])
    print("Status:", result["status"])

    print("\nREAL DATA TEST COMPLETED")


if __name__ == "__main__":
    run_real_data_test()
