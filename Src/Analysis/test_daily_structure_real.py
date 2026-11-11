
import csv
from pathlib import Path
from datetime import datetime, timezone

from Src.Analysis.daily_structure import analyze_daily_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Archive"
    / "BTC_USDT_SPOT_1D.csv"
)


def load_real_candles():
    if not ARCHIVE_FILE.exists():
        raise FileNotFoundError(
            f"Daily archive not found: {ARCHIVE_FILE}"
        )

    candles = []

    with open(ARCHIVE_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("Archive CSV has missing required columns.")

        for row in reader:
            candles.append({
                "timestamp": int(float(row["timestamp"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            })

    if not candles:
        raise ValueError("Daily archive contains no candles.")

    return candles


def format_timestamp(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")


def print_structure(name, structure):
    print(f"\n{name}")
    print("-" * 40)
    print("Trend:", structure["trend"])

    if structure["trend"] == "UNKNOWN":
        print("Reason:", structure["reason"])
        return

    print(
        "Previous High:",
        structure["previous_high"]["price"],
        format_timestamp(structure["previous_high"]["timestamp"])
    )

    print(
        "Latest High:",
        structure["latest_high"]["price"],
        format_timestamp(structure["latest_high"]["timestamp"])
    )

    print(
        "Previous Low:",
        structure["previous_low"]["price"],
        format_timestamp(structure["previous_low"]["timestamp"])
    )

    print(
        "Latest Low:",
        structure["latest_low"]["price"],
        format_timestamp(structure["latest_low"]["timestamp"])
    )


def run_real_data_test():
    candles = load_real_candles()

    print("REAL OKX DAILY STRUCTURE TEST")
    print("=" * 50)
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

    print("\nFINAL RESULT")
    print("=" * 50)
    print("Allowed Direction:", result["allowed_direction"])
    print("Analysis Status:", result["status"])
    print("\nREAL DATA TEST COMPLETED")


if __name__ == "__main__":
    run_real_data_test()
