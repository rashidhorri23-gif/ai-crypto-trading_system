
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from Src.Analysis.daily_structure import (
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)

from Src.Analysis.four_hour_setup import generate_four_hour_setup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Archive"
    / "BTC_USDT_SPOT_4H.csv"
)

VALID_EVENTS = {
    "BOS_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BULLISH",
    "CHOCH_BEARISH",
}


def parse_timestamp(value):
    value = str(value).strip()

    try:
        return float(value)
    except ValueError:
        pass

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.timestamp() * 1000


def format_timestamp(value):
    if value is None:
        return "N/A"

    timestamp = int(float(value))

    if timestamp > 100000000000:
        timestamp //= 1000

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_candles(filepath):
    if not filepath.exists():
        raise FileNotFoundError(
            f"4H archive file not found: {filepath}"
        )

    candles = []

    with filepath.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file has no header.")

        column_map = {
            name.strip().lower(): name
            for name in reader.fieldnames
        }

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        }

        missing = required - set(column_map)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                candle = {
                    "timestamp": parse_timestamp(
                        row[column_map["timestamp"]]
                    ),
                    "open": float(row[column_map["open"]]),
                    "high": float(row[column_map["high"]]),
                    "low": float(row[column_map["low"]]),
                    "close": float(row[column_map["close"]]),
                }

                if "volume" in column_map:
                    candle["volume"] = float(
                        row[column_map["volume"]]
                    )

                candles.append(candle)

            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(
                    f"Invalid CSV row {row_number}: {error}"
                ) from error

    if len(candles) < 20:
        raise ValueError(
            "Not enough candles for historical analysis."
        )

    return candles


def event_direction(event):
    if event["direction"] == "BULLISH":
        return "LONG"

    if event["direction"] == "BEARISH":
        return "SHORT"

    return None


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def collect_diagnostics(diagnostics, diagnostic_counts, rejection_counts):
    if not isinstance(diagnostics, dict):
        return

    for key, value in diagnostics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            diagnostic_counts[key] += value

        elif isinstance(value, list):
            diagnostic_counts[f"{key}_items"] += len(value)

            if key == "rejected_candidates":
                for item in value:
                    if isinstance(item, dict):
                        reason = item.get(
                            "reason",
                            "UNKNOWN_REJECTION_REASON"
                        )
                        rejection_counts[str(reason)] += 1


def run_historical_confluence_audit():
    print_section("HISTORICAL 4H CONFLUENCE AUDIT")

    print(f"Archive: {ARCHIVE_PATH}")

    candles = load_candles(ARCHIVE_PATH)

    print(f"Total Candles: {len(candles)}")
    print(
        "First Candle:",
        format_timestamp(candles[0]["timestamp"]),
    )
    print(
        "Last Candle:",
        format_timestamp(candles[-1]["timestamp"]),
    )

    candle_duration = infer_candle_duration(candles)

    if candle_duration <= 0:
        raise ValueError("Invalid 4H candle duration.")

    swings = detect_swings(
        candles,
        left_bars=3,
        right_bars=3,
        candle_duration=candle_duration,
    )

    events = analyze_structure_events(
        candles,
        swings,
        candle_duration,
    )

    events = [
        event
        for event in events
        if (
            isinstance(event, dict)
            and event.get("event") in VALID_EVENTS
            and isinstance(event.get("index"), int)
            and 0 <= event["index"] < len(candles)
            and event.get("direction") in ("BULLISH", "BEARISH")
            and isinstance(event.get("available_timestamp"), (int, float))
        )
    ]

    events.sort(
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
        )
    )

    print(f"Confirmed Structure Events: {len(events)}")

    status_counts = Counter()
    reason_counts = Counter()
    direction_counts = Counter()
    event_type_counts = Counter()
    diagnostic_counts = Counter()
    rejection_counts = Counter()

    confirmed_setups = []
    watch_setups = []
    rejected_events = []

    for event_number, event in enumerate(events, start=1):
        event_index = event["index"]
        direction = event_direction(event)

        if direction is None:
            rejected_events.append({
                "event_number": event_number,
                "reason": "INVALID_DIRECTION",
            })
            continue

        historical_candles = candles[:event_index + 1]

        try:
            result = generate_four_hour_setup(
                candles=historical_candles,
                allowed_direction=direction,
                setup_expiry_candles=24,
                ob_lookback=10,
                fvg_lookback=10,
                risk_reward=2.0,
                symbol="BTC-USDT",
            )

        except Exception as error:
            rejected_events.append({
                "event_number": event_number,
                "event": event["event"],
                "timestamp": event["timestamp"],
                "reason": f"ENGINE_ERROR: {error}",
            })
            continue

        diagnostics = result.get("diagnostics", {})

        collect_diagnostics(
            diagnostics,
            diagnostic_counts,
            rejection_counts,
        )

        returned_setup = result.get("setup")

        if returned_setup is not None:
            same_event = (
                returned_setup.get("structure_event")
                == event["event"]
                and returned_setup.get(
                    "structure_event_timestamp"
                ) == event["timestamp"]
            )

            if not same_event:
                rejected_events.append({
                    "event_number": event_number,
                    "event": event["event"],
                    "timestamp": event["timestamp"],
                    "reason": "ENGINE_EVENT_MISMATCH",
                    "returned_event": returned_setup.get(
                        "structure_event"
                    ),
                    "returned_timestamp": returned_setup.get(
                        "structure_event_timestamp"
                    ),
                })
                continue

        status = result.get("status", "UNKNOWN")
        reason = result.get("reason", "No reason returned.")

        status_counts[status] += 1
        reason_counts[reason] += 1
        direction_counts[direction] += 1
        event_type_counts[event["event"]] += 1

        record = {
            "event_number": event_number,
            "event": event["event"],
            "direction": direction,
            "timestamp": event["timestamp"],
            "available_timestamp": event["available_timestamp"],
            "status": status,
            "reason": reason,
            "diagnostics": diagnostics,
        }

        if returned_setup is not None:
            record["setup_id"] = returned_setup.get("setup_id")
            record["entry_zone"] = returned_setup.get("entry_zone")
            record["stop_loss"] = returned_setup.get("stop_loss")
            record["take_profit"] = returned_setup.get("take_profit")
            record["order_block"] = returned_setup.get("order_block")
            record["fvg"] = returned_setup.get("fvg")
            record["zone_overlap"] = returned_setup.get("zone_overlap")

        if status == "CONFIRMED_CONFLUENCE":
            confirmed_setups.append(record)

        elif status == "WATCH_NO_ZONE_OVERLAP":
            watch_setups.append(record)

    print_section("HISTORICAL SUMMARY")

    print(f"Processed Events: {sum(status_counts.values())}")
    print(f"Confirmed Confluence: {status_counts['CONFIRMED_CONFLUENCE']}")
    print(f"Watch Without Overlap: {status_counts['WATCH_NO_ZONE_OVERLAP']}")
    print(f"Wait / Rejected by Rules: {status_counts['WAIT']}")
    print(f"Other Statuses: {sum(status_counts.values()) - status_counts['CONFIRMED_CONFLUENCE'] - status_counts['WATCH_NO_ZONE_OVERLAP'] - status_counts['WAIT']}")
    print(f"Engine Errors / Mismatches: {len(rejected_events)}")

    print_section("DIRECTION DISTRIBUTION")

    for direction, count in sorted(direction_counts.items()):
        print(f"{direction}: {count}")

    print_section("STRUCTURE EVENT DISTRIBUTION")

    for event_type, count in sorted(event_type_counts.items()):
        print(f"{event_type}: {count}")

    print_section("ENGINE DIAGNOSTIC TOTALS")

    if not diagnostic_counts:
        print("No numeric or list diagnostics were returned.")
    else:
        for key, count in sorted(diagnostic_counts.items()):
            print(f"{key}: {count}")

    print_section("CANDIDATE REJECTION REASONS")

    if not rejection_counts:
        print("No rejected candidate details were returned.")
    else:
        for reason, count in rejection_counts.most_common():
            print(f"{count} | {reason}")

    print_section("GENERAL ENGINE REASONS")

    for reason, count in reason_counts.most_common():
        print(f"{count} | {reason}")

    print_section("CONFIRMED CONFLUENCE SETUPS")

    if not confirmed_setups:
        print("No confirmed confluence setups found.")

    for record in confirmed_setups:
        print()
        print(f"Event Number: {record['event_number']}")
        print(f"Event: {record['event']}")
        print(f"Timestamp: {format_timestamp(record['timestamp'])}")
        print(f"Setup ID: {record.get('setup_id')}")
        print(f"Entry Zone: {record.get('entry_zone')}")
        print(f"Stop Loss: {record.get('stop_loss')}")
        print(f"Take Profit: {record.get('take_profit')}")
        print(f"Order Block: {record.get('order_block')}")
        print(f"FVG: {record.get('fvg')}")
        print(f"Overlap: {record.get('zone_overlap')}")

    print_section("WATCH SETUPS WITHOUT OVERLAP")

    if not watch_setups:
        print("No watch-only setups found.")

    for record in watch_setups:
        print()
        print(f"Event Number: {record['event_number']}")
        print(f"Event: {record['event']}")
        print(f"Direction: {record['direction']}")
        print(f"Timestamp: {format_timestamp(record['timestamp'])}")
        print(f"Setup ID: {record.get('setup_id')}")
        print(f"Entry Zone: {record.get('entry_zone')}")
        print(f"Order Block: {record.get('order_block')}")
        print(f"FVG: {record.get('fvg')}")

    print_section("ENGINE ERRORS AND EVENT MISMATCHES")

    if not rejected_events:
        print("No engine errors or event mismatches.")

    for record in rejected_events:
        print(record)

    print_section("AUDIT COMPLETED")

    print("Historical analysis only.")
    print("No trades were executed.")
    print("Live Trading: DISABLED")


if __name__ == "__main__":
    run_historical_confluence_audit()
