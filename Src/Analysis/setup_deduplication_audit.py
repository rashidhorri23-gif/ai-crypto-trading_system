
from collections import Counter

from Src.Analysis.confluence_history_audit import (
    ARCHIVE_PATH,
    VALID_EVENTS,
    load_candles,
    event_direction,
    format_timestamp,
)

from Src.Analysis.daily_structure import (
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)

from Src.Analysis.four_hour_setup import generate_four_hour_setup

from Src.Analysis.setup_deduplication import (
    deduplicate_setups,
    print_deduplication_report,
)


def run_setup_deduplication_audit():
    print()
    print("=" * 72)
    print("INDEPENDENT 4H SETUP DEDUPLICATION AUDIT")
    print("=" * 72)

    print("Archive:", ARCHIVE_PATH)

    candles = load_candles(ARCHIVE_PATH)

    print("Total Candles:", len(candles))
    print("First Candle:", format_timestamp(candles[0]["timestamp"]))
    print("Last Candle:", format_timestamp(candles[-1]["timestamp"]))

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
            and not isinstance(event.get("index"), bool)
            and 0 <= event["index"] < len(candles)
            and event.get("direction") in ("BULLISH", "BEARISH")
            and isinstance(
                event.get("available_timestamp"),
                (int, float),
            )
            and not isinstance(
                event.get("available_timestamp"),
                bool,
            )
        )
    ]

    events.sort(
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
        )
    )

    print("Confirmed Structure Events:", len(events))

    confirmed_setups = []
    status_counts = Counter()
    engine_errors = []
    event_mismatches = []

    for event_number, event in enumerate(events, start=1):
        direction = event_direction(event)

        if direction is None:
            continue

        historical_candles = candles[:event["index"] + 1]

        try:
            result = generate_four_hour_setup(
                candles=historical_candles,
                allowed_direction=direction,
                setup_expiry_candles=24,
                ob_lookback=10,
                fvg_lookback=10,
                risk_reward=2.0,
                symbol="BTC-USDT",
                target_event=event,
            )

        except Exception as error:
            engine_errors.append({
                "event_number": event_number,
                "event": event.get("event"),
                "error": "{}: {}".format(
                    type(error).__name__,
                    error,
                ),
            })
            continue

        diagnostics = result.get("diagnostics", {})

        if diagnostics.get("matching_events") != 1:
            event_mismatches.append({
                "event_number": event_number,
                "event": event.get("event"),
                "timestamp": event.get("timestamp"),
                "matching_events": diagnostics.get(
                    "matching_events"
                ),
            })
            continue

        status = result.get("status", "UNKNOWN")
        status_counts[status] += 1

        setup = result.get("setup")

        if setup is None:
            continue

        same_event = (
            setup.get("structure_event") == event.get("event")
            and setup.get("structure_event_timestamp")
            == event.get("timestamp")
        )

        if not same_event:
            event_mismatches.append({
                "event_number": event_number,
                "event": event.get("event"),
                "reason": "RETURNED_SETUP_EVENT_MISMATCH",
                "returned_event": setup.get("structure_event"),
                "returned_timestamp": setup.get(
                    "structure_event_timestamp"
                ),
            })
            continue

        if status != "CONFIRMED_CONFLUENCE":
            continue

        confirmed_setups.append(setup)

    print()
    print("-" * 72)
    print("RAW HISTORICAL SETUP SUMMARY")
    print("-" * 72)

    print("Confirmed Setups Before Deduplication:", len(confirmed_setups))

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    print("Engine Errors:", len(engine_errors))
    print("Event Mismatches:", len(event_mismatches))

    deduplication_result = deduplicate_setups(
        confirmed_setups,
        default_symbol="BTC-USDT",
    )

    print_deduplication_report(
        deduplication_result,
        title="INDEPENDENT 4H STRUCTURAL DEDUPLICATION REPORT",
    )

    print()
    print("-" * 72)
    print("RETAINED UNIQUE SETUPS")
    print("-" * 72)

    for number, setup in enumerate(
        deduplication_result["retained_setups"],
        start=1,
    ):
        order_block = setup.get("order_block", {})
        fvg = setup.get("fvg", {})

        print()
        print(f"Setup #{number}")
        print("Setup ID:", setup.get("setup_id"))
        print("Direction:", setup.get("direction"))
        print("Structure Event:", setup.get("structure_event"))
        print(
            "Available:",
            format_timestamp(setup.get("available_timestamp")),
        )
        print("OB Timestamp:", order_block.get("timestamp"))
        print("FVG Timestamp:", fvg.get("timestamp"))
        print("Entry Zone:", setup.get("entry_zone"))
        print("Stop Loss:", setup.get("stop_loss"))
        print("Take Profit:", setup.get("take_profit"))

    print()
    print("-" * 72)
    print("ENGINE ERRORS")
    print("-" * 72)

    if not engine_errors:
        print("No engine errors.")

    for item in engine_errors:
        print(item)

    print()
    print("-" * 72)
    print("EVENT MISMATCHES")
    print("-" * 72)

    if not event_mismatches:
        print("No event mismatches.")

    for item in event_mismatches:
        print(item)

    print()
    print("=" * 72)
    print("INDEPENDENT AUDIT COMPLETED")
    print("=" * 72)
    print("Historical analysis only.")
    print("No trades were executed.")
    print("Live Trading: DISABLED")


if __name__ == "__main__":
    run_setup_deduplication_audit()
