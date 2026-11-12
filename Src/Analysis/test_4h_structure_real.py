
import csv
from datetime import datetime, timezone
from pathlib import Path

from Src.Analysis.daily_structure import analyze_daily_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Archive"
    / "BTC_USDT_SPOT_4H.csv"
)


def format_timestamp(timestamp):
    if timestamp is None:
        return "N/A"

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
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=2):
            try:
                candles.append({
                    "timestamp": int(float(row["timestamp"])),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume") or 0),
                })
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Invalid CSV row {row_number}: {error}"
                ) from error

    return candles[-540:]


def print_swing(title, swing):
    if not swing:
        print(f"{title}: NONE")
        return

    print(
        f"{title} | "
        f"Type: {swing.get('type')} | "
        f"Price: {swing.get('price')} | "
        f"Pivot: {format_timestamp(swing.get('timestamp'))} | "
        f"Confirmed: {format_timestamp(swing.get('confirmation_timestamp'))} | "
        f"Index: {swing.get('index')} | "
        f"Confirmation Index: {swing.get('confirmation_index')}"
    )


def print_structure(title, result):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    print("Trend:", result.get("trend"))
    print("Structure Reason:", result.get("reason"))
    print("High Structure:", result.get("high_structure"))
    print("Low Structure:", result.get("low_structure"))
    print("Event Bias:", result.get("event_bias"))

    print("Swing High Count:", result.get("swing_high_count"))
    print("Swing Low Count:", result.get("swing_low_count"))

    print_swing(
        "Latest Swing High",
        result.get("latest_swing_high")
    )

    print_swing(
        "Latest Swing Low",
        result.get("latest_swing_low")
    )

    swings = result.get("swings", [])

    print("\nConfirmed Swings:", len(swings))

    for swing in swings[-10:]:
        print(
            f"{swing.get('type')} | "
            f"Price: {swing.get('price')} | "
            f"Pivot: {format_timestamp(swing.get('timestamp'))} | "
            f"Available: {format_timestamp(swing.get('confirmation_timestamp'))} | "
            f"Index: {swing.get('index')} | "
            f"Confirmation Index: {swing.get('confirmation_index')}"
        )

    events = result.get("events", [])

    print("\nStructure Events:", len(events))

    for event in events:
        print(
            f"{event.get('event')} | "
            f"Direction: {event.get('direction')} | "
            f"Level: {event.get('level', 'N/A')} | "
            f"Time: {format_timestamp(event.get('timestamp'))} | "
            f"Available: {format_timestamp(event.get('available_timestamp'))} | "
            f"Broken Swing: {event.get('broken_swing_type')} | "
            f"Broken Swing Index: {event.get('broken_swing_index')}"
        )


def run_real_4h_test():
    print("REAL OKX 4H STRUCTURE TEST")
    print("=" * 72)

    candles = load_candles()

    if len(candles) < 20:
        raise ValueError(
            f"Not enough 4H candles: {len(candles)}"
        )

    print("Source:", ARCHIVE_FILE.name)
    print("Total candles:", len(candles))
    print("First candle:", format_timestamp(candles[0]["timestamp"]))
    print("Last candle:", format_timestamp(candles[-1]["timestamp"]))
    print("Latest close:", candles[-1]["close"])

    result = analyze_daily_structure(
        candles,
        timeframe="4H"
    )

    print_structure(
        "4H INTERNAL STRUCTURE",
        result["internal_structure"]
    )

    print_structure(
        "4H EXTERNAL STRUCTURE",
        result["external_structure"]
    )

    print("\n" + "=" * 72)
    print("FINAL 4H DECISION")
    print("=" * 72)

    print("Combined Trend:", result.get("trend"))
    print("External Event Bias:", result.get("event_bias"))
    print("Allowed Direction:", result.get("allowed_direction"))
    print("Directions to Evaluate:", result.get("trade_directions_to_evaluate"))
    print("Status:", result.get("status"))

    print("\nREAL 4H DATA TEST COMPLETED")


if __name__ == "__main__":
    run_real_4h_test()
