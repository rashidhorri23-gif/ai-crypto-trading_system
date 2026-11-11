

import csv
from datetime import datetime, timezone
from pathlib import Path

from Src.Analysis.daily_structure import analyze_daily_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"
REPORT_DIR = PROJECT_ROOT / "Reports"

FILES = {
    "1D": "BTC_USDT_SPOT_1D.csv",
    "4H": "BTC_USDT_SPOT_4H.csv",
}

FAILED_BREAK_LOOKAHEAD = 3


def parse_timestamp(value):
    value = str(value).strip()

    try:
        number = float(value)

        if number > 100000000000:
            number = number / 1000

        return int(number)

    except ValueError:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return int(parsed.timestamp())


def load_candles(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archive file not found: {file_path}"
        )

    candles = []

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV header is missing.")

        normalized_headers = {
            name.strip().lower(): name
            for name in reader.fieldnames
        }

        required = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            field for field in required
            if field not in normalized_headers
        ]

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        for row in reader:
            candle = {
                "timestamp": parse_timestamp(
                    row[normalized_headers["timestamp"]]
                ),
                "open": float(
                    row[normalized_headers["open"]]
                ),
                "high": float(
                    row[normalized_headers["high"]]
                ),
                "low": float(
                    row[normalized_headers["low"]]
                ),
                "close": float(
                    row[normalized_headers["close"]]
                ),
                "volume": float(
                    row[normalized_headers["volume"]]
                ),
            }

            candles.append(candle)

    candles.sort(key=lambda item: item["timestamp"])

    if not candles:
        raise ValueError(
            f"No candles found in archive: {file_path}"
        )

    return candles


def format_time(timestamp):
    return datetime.fromtimestamp(
        int(timestamp),
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_price(value):
    if value is None:
        return "N/A"

    return f"{float(value):.4f}"


def audit_failed_breaks(events, candles):
    results = []

    for event in events:
        event_type = event["event"]

        if event_type not in (
            "BOS_BULLISH",
            "CHOCH_BULLISH",
            "BOS_BEARISH",
            "CHOCH_BEARISH",
        ):
            continue

        level = float(event["level"])
        event_index = int(event["index"])

        future_candles = candles[
            event_index + 1:
            event_index + 1 + FAILED_BREAK_LOOKAHEAD
        ]

        returned = False
        return_index = None
        return_close = None

        for offset, candle in enumerate(future_candles, start=1):
            close = float(candle["close"])

            if event_type.endswith("BULLISH"):
                if close <= level:
                    returned = True
                    return_index = event_index + offset
                    return_close = close
                    break

            elif event_type.endswith("BEARISH"):
                if close >= level:
                    returned = True
                    return_index = event_index + offset
                    return_close = close
                    break

        if returned:
            status = "RETURNED_TO_LEVEL_WITHIN_3_CANDLES"
        elif len(future_candles) < FAILED_BREAK_LOOKAHEAD:
            status = "INSUFFICIENT_FUTURE_DATA"
        else:
            status = "HELD_BEYOND_LEVEL_FOR_3_CANDLES"

        results.append({
            "event": event_type,
            "level": level,
            "timestamp": event["timestamp"],
            "status": status,
            "return_timestamp": (
                candles[return_index]["timestamp"]
                if return_index is not None
                else None
            ),
            "return_close": return_close,
        })

    return results


def add_structure_report(
    lines,
    timeframe,
    structure_name,
    result,
    candles,
    candle_duration_seconds
):
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"{timeframe} {structure_name} STRUCTURE")
    lines.append("=" * 70)

    lines.append(f"Trend: {result['trend']}")
    lines.append(f"Event Bias: {result['event_bias']}")
    lines.append(f"Swing High Count: {result['high_count']}")
    lines.append(f"Swing Low Count: {result['low_count']}")

    lines.append("")
    lines.append("CONFIRMED SWINGS")
    lines.append("-" * 70)

    for swing in result["swings"]:
        confirmation_time = int(
            swing["confirmation_timestamp"]
        )

        available_time = (
            confirmation_time + candle_duration_seconds
        )

        lines.append(
            f"{swing['type']} | "
            f"Price: {format_price(swing['price'])} | "
            f"Pivot: {format_time(swing['timestamp'])} | "
            f"Confirmed: {format_time(confirmation_time)} | "
            f"Available: {format_time(available_time)} | "
            f"Index: {swing['index']}"
        )

    lines.append("")
    lines.append("STRUCTURE EVENTS")
    lines.append("-" * 70)

    for event in result["events"]:
        event_name = event["event"]

        if event_name == "AMBIGUOUS_DOUBLE_BREAK":
            lines.append(
                f"{event_name} | "
                f"Time: {format_time(event['timestamp'])} | "
                f"Bullish Level: {format_price(event['bullish_level'])} | "
                f"Bearish Level: {format_price(event['bearish_level'])}"
            )
            continue

        lines.append(
            f"{event_name} | "
            f"Level: {format_price(event['level'])} | "
            f"Close: {format_price(event['close'])} | "
            f"Time: {format_time(event['timestamp'])} | "
            f"Available: {format_time(event['available_timestamp'])}"
        )

    lines.append("")
    lines.append("BREAK RETEST AUDIT")
    lines.append("-" * 70)

    failed_results = audit_failed_breaks(
        result["events"],
        candles
    )

    for item in failed_results:
        line = (
            f"{item['event']} | "
            f"Level: {format_price(item['level'])} | "
            f"Status: {item['status']}"
        )

        if item["return_timestamp"] is not None:
            line += (
                f" | Return Time: "
                f"{format_time(item['return_timestamp'])}"
                f" | Return Close: "
                f"{format_price(item['return_close'])}"
            )

        lines.append(line)

    if not failed_results:
        lines.append("No directional break events found.")


def generate_report(timeframe, filename):
    file_path = ARCHIVE_DIR / filename
    candles = load_candles(file_path)

    result = analyze_daily_structure(
        candles,
        timeframe=timeframe
    )

    candle_duration_seconds = int(
        result["candle_duration_seconds"]
    )

    lines = []

    lines.append("=" * 70)
    lines.append("REAL OKX STRUCTURE AUDIT REPORT")
    lines.append("=" * 70)

    lines.append(f"Source: {filename}")
    lines.append(f"Timeframe: {timeframe}")
    lines.append(
        f"Candle Duration: {candle_duration_seconds} seconds"
    )
    lines.append(f"Total Candles: {len(candles)}")
    lines.append(
        f"First Candle: {format_time(candles[0]['timestamp'])}"
    )
    lines.append(
        f"Last Candle: {format_time(candles[-1]['timestamp'])}"
    )
    lines.append(
        f"Latest Close: {format_price(candles[-1]['close'])}"
    )

    lines.append(
        f"Internal Bias: "
        f"{result['internal_structure']['event_bias']}"
    )

    lines.append(
        f"External Bias: "
        f"{result['external_structure']['event_bias']}"
    )

    lines.append(
        f"Directions Evaluated: "
        f"{', '.join(result['trade_directions_to_evaluate'])}"
    )

    add_structure_report(
        lines,
        timeframe,
        "INTERNAL",
        result["internal_structure"],
        candles,
        candle_duration_seconds
    )

    add_structure_report(
        lines,
        timeframe,
        "EXTERNAL",
        result["external_structure"],
        candles,
        candle_duration_seconds
    )

    return "\n".join(lines)


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_reports = []

    for timeframe, filename in FILES.items():
        report = generate_report(
            timeframe,
            filename
        )

        all_reports.append(report)

        print(report)
        print("")

    combined_report = "\n\n".join(all_reports)

    output_file = (
        REPORT_DIR / "BTC_Structure_Audit_Report.txt"
    )

    output_file.write_text(
        combined_report,
        encoding="utf-8"
    )

    print("=" * 70)
    print("STRUCTURE AUDIT COMPLETED")
    print(f"Report: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
