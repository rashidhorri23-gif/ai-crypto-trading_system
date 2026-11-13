
import json
from collections import Counter

from Src.Main import (
    SYMBOL,
    TIMEFRAME_CONFIG,
    load_structure_candles,
)

from Src.Analysis.daily_structure import (
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)

from Src.Analysis.four_hour_setup import (
    generate_four_hour_setup,
)


MODES = ("A", "B", "C")
VALID_EVENTS = {
    "BOS_BULLISH",
    "CHOCH_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BEARISH",
}


def print_section(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def json_print(value):
    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )


def get_event_direction(event):
    direction = event.get("direction")

    if direction == "BULLISH":
        return "LONG"

    if direction == "BEARISH":
        return "SHORT"

    return None


def get_event_key(event):
    return (
        event.get("event"),
        event.get("index"),
        event.get("timestamp"),
    )


def run_four_hour_event_rejection_audit(candles, symbol=SYMBOL):
    print_section("FOUR HOUR EVENT REJECTION AUDIT")

    print("Symbol:", symbol)
    print("Timeframe: 4H")
    print("Historical Candles:", len(candles))
    print("Modes:", ", ".join(MODES))
    print("Directions: LONG / SHORT")
    print("Lookahead: DISABLED")
    print("Live Trading: DISABLED")
    print("Trade Execution: DISABLED")

    if len(candles) < 100:
        raise RuntimeError(
            "At least 100 candles are required for the audit."
        )

    candles = validate_candles(list(candles))
    duration = infer_candle_duration(candles)

    swings = detect_swings(
        candles,
        left_bars=3,
        right_bars=3,
        candle_duration=duration,
    )

    events = analyze_structure_events(
        candles,
        swings,
        duration,
    )

    matching_events = []

    for event in events:
        if event.get("event") not in VALID_EVENTS:
            continue

        direction = get_event_direction(event)

        if direction is None:
            continue

        index = event.get("index")
        timestamp = event.get("timestamp")
        available_timestamp = event.get("available_timestamp")

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(candles)
        ):
            continue

        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
        ):
            continue

        if (
            isinstance(available_timestamp, bool)
            or not isinstance(available_timestamp, int)
        ):
            continue

        if candles[index]["timestamp"] != timestamp:
            continue

        if available_timestamp != timestamp + duration:
            continue

        matching_events.append(event)

    matching_events.sort(
        key=lambda item: (
            item["available_timestamp"],
            item["index"],
        )
    )

    print("\nINPUT VALIDATION")
    print("----------------")
    print("Candle Duration:", duration)
    print("Swing Count:", len(swings))
    print("All Structure Events:", len(events))
    print("Valid Target Events:", len(matching_events))

    if not matching_events:
        print("\nNo valid structure events were found.")
        print("Audit completed without setup evaluation.")
        return {
            "events": 0,
            "results": {},
        }

    summary = {
        mode: {
            "evaluated": 0,
            "confirmed": 0,
            "expired": 0,
            "rejected": 0,
            "engine_errors": 0,
            "rejection_reasons": Counter(),
            "pair_statuses": Counter(),
            "ob_candidates_examined": 0,
            "valid_ob_candidates": 0,
            "fvg_patterns_checked": 0,
            "valid_fvg_patterns": 0,
            "ob_fvg_pairs_checked": 0,
            "overlap_confirmed": 0,
            "overlap_failed": 0,
            "mitigated": 0,
            "valid_stop_swings": 0,
        }
        for mode in MODES
    }

    detailed_results = []

    for event_number, event in enumerate(matching_events, start=1):
        direction = get_event_direction(event)
        event_index = event["index"]

        target_event = {
            "event": event["event"],
            "index": event_index,
            "timestamp": event["timestamp"],
        }

        print_section(
            "STRUCTURE EVENT {}/{}".format(
                event_number,
                len(matching_events),
            )
        )

        print("Event:", event["event"])
        print("Direction:", direction)
        print("Event Index:", event_index)
        print("Event Timestamp:", event["timestamp"])
        print("Available Timestamp:", event["available_timestamp"])
        print("Broken Swing Index:", event.get("broken_swing_index"))
        print("Broken Swing Type:", event.get("broken_swing_type"))
        print("Broken Level:", event.get("level"))

        # Evaluate the event at its first available candle close.
        # Later candles are intentionally excluded.
        event_candles = candles[:event_index + 1]

        if not event_candles:
            continue

        for mode in MODES:
            print_section(
                "EVENT {} | MODE {}".format(
                    event_number,
                    mode,
                )
            )

            try:
                result = generate_four_hour_setup(
                    candles=event_candles,
                    allowed_direction=direction,
                    setup_expiry_candles=24,
                    ob_lookback=10,
                    fvg_lookback=10,
                    risk_reward=2.0,
                    symbol=symbol,
                    target_event=target_event,
                    mitigation_mode=mode,
                )

                diagnostics = result.get("diagnostics", {})
                setup = result.get("setup")

                record = {
                    "event": event["event"],
                    "direction": direction,
                    "event_index": event_index,
                    "event_timestamp": event["timestamp"],
                    "mode": mode,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "diagnostics": diagnostics,
                }

                detailed_results.append(record)

                stats = summary[mode]
                stats["evaluated"] += 1

                if isinstance(setup, dict):
                    stats["confirmed"] += 1
                else:
                    expired_count = diagnostics.get(
                        "expired_candidate_count",
                        0,
                    )

                    rejected_count = diagnostics.get(
                        "rejected_candidate_count",
                        0,
                    )

                    if expired_count:
                        stats["expired"] += 1

                    if rejected_count:
                        stats["rejected"] += 1

                for key in (
                    "ob_candidates_examined",
                    "ob_valid_candidates",
                    "fvg_patterns_checked",
                    "fvg_valid_patterns",
                    "ob_fvg_pairs_checked",
                    "ob_fvg_overlap_confirmed",
                    "ob_fvg_overlap_failed",
                    "ob_mitigated",
                    "valid_stop_swings",
                ):
                    mapped_key = {
                        "ob_candidates_examined": "ob_candidates_examined",
                        "ob_valid_candidates": "valid_ob_candidates",
                        "fvg_patterns_checked": "fvg_patterns_checked",
                        "fvg_valid_patterns": "valid_fvg_patterns",
                        "ob_fvg_pairs_checked": "ob_fvg_pairs_checked",
                        "ob_fvg_overlap_confirmed": "overlap_confirmed",
                        "ob_fvg_overlap_failed": "overlap_failed",
                        "ob_mitigated": "mitigated",
                        "valid_stop_swings": "valid_stop_swings",
                    }[key]

                    stats[mapped_key] += diagnostics.get(key, 0)

                rejected_candidates = diagnostics.get(
                    "rejected_candidates",
                    [],
                )

                expired_candidates = diagnostics.get(
                    "expired_candidates",
                    [],
                )

                pair_details = diagnostics.get(
                    "ob_fvg_pair_details",
                    [],
                )

                for rejected in rejected_candidates:
                    reason = rejected.get(
                        "reason",
                        "UNKNOWN",
                    )
                    stats["rejection_reasons"][reason] += 1

                for pair in pair_details:
                    stats["pair_statuses"][
                        pair.get("status", "UNKNOWN")
                    ] += 1

                print("Status:", result.get("status"))
                print("Reason:", result.get("reason"))
                print(
                    "Setup Generated:",
                    isinstance(setup, dict),
                )

                print("\nKEY DIAGNOSTICS")
                print("----------------")
                for key in (
                    "matching_events",
                    "evaluated_candidate_count",
                    "expired_candidate_count",
                    "rejected_candidate_count",
                    "order_block_search_attempts",
                    "ob_candidates_examined",
                    "ob_wrong_candle_direction",
                    "ob_valid_candidates",
                    "fvg_patterns_checked",
                    "fvg_no_gap",
                    "fvg_valid_patterns",
                    "ob_fvg_pairs_checked",
                    "ob_mitigated",
                    "ob_fvg_overlap_confirmed",
                    "ob_fvg_overlap_failed",
                    "valid_stop_swings",
                ):
                    print(
                        "{}: {}".format(
                            key,
                            diagnostics.get(key, 0),
                        )
                    )

                print("\nREJECTION DETAILS")
                print("-----------------")
                if rejected_candidates:
                    json_print(rejected_candidates)
                else:
                    print("No rejected candidates recorded.")

                print("\nEXPIRY DETAILS")
                print("--------------")
                if expired_candidates:
                    json_print(expired_candidates)
                else:
                    print("No expired candidates recorded.")

                print("\nOB-FVG PAIR DETAILS")
                print("-------------------")
                if pair_details:
                    json_print(pair_details)
                else:
                    print("No OB/FVG pairs were evaluated.")

                if isinstance(setup, dict):
                    print("\nCONFIRMED SETUP")
                    print("----------------")
                    json_print(setup)

            except Exception as error:
                summary[mode]["engine_errors"] += 1

                error_record = {
                    "event": event["event"],
                    "direction": direction,
                    "event_index": event_index,
                    "mode": mode,
                    "error": "{}: {}".format(
                        type(error).__name__,
                        error,
                    ),
                }

                detailed_results.append(error_record)

                print("ENGINE ERROR")
                print("------------")
                print(error_record)

    print_section("FINAL EVENT REJECTION SUMMARY")

    print("Total Valid Structure Events:", len(matching_events))
    print("Expected Evaluations:", len(matching_events) * len(MODES))

    for mode in MODES:
        stats = summary[mode]

        print_section("MODE {} SUMMARY".format(mode))

        print("Evaluated Events:", stats["evaluated"])
        print("Confirmed Setups:", stats["confirmed"])
        print("Expired Events:", stats["expired"])
        print("Rejected Events:", stats["rejected"])
        print("Engine Errors:", stats["engine_errors"])

        print("\nORDER BLOCK SUMMARY")
        print("-------------------")
        print(
            "OB Candidates Examined:",
            stats["ob_candidates_examined"],
        )
        print(
            "Valid OB Candidates:",
            stats["valid_ob_candidates"],
        )
        print("Mitigated OBs:", stats["mitigated"])

        print("\nFVG SUMMARY")
        print("-----------")
        print(
            "FVG Patterns Checked:",
            stats["fvg_patterns_checked"],
        )
        print(
            "Valid FVG Patterns:",
            stats["valid_fvg_patterns"],
        )

        print("\nOB-FVG SUMMARY")
        print("--------------")
        print(
            "OB/FVG Pairs Checked:",
            stats["ob_fvg_pairs_checked"],
        )
        print(
            "Confirmed Overlaps:",
            stats["overlap_confirmed"],
        )
        print(
            "Failed Overlaps:",
            stats["overlap_failed"],
        )
        print(
            "Valid Stop Swings:",
            stats["valid_stop_swings"],
        )

        print("\nREJECTION REASONS")
        print("-----------------")
        if stats["rejection_reasons"]:
            for reason, count in stats["rejection_reasons"].most_common():
                print("{}: {}".format(reason, count))
        else:
            print("No rejection reasons recorded.")

        print("\nOB-FVG PAIR STATUSES")
        print("--------------------")
        if stats["pair_statuses"]:
            for status, count in stats["pair_statuses"].most_common():
                print("{}: {}".format(status, count))
        else:
            print("No pair statuses recorded.")

    print_section("AUDIT COMPLETED")
    print("Detailed Event Evaluations:", len(detailed_results))
    print("No trades were executed.")
    print("Live Trading: DISABLED")

    return {
        "events": len(matching_events),
        "summary": summary,
        "results": detailed_results,
    }


def main():
    config = TIMEFRAME_CONFIG["4H"]

    print_section("LOADING 4H ARCHIVE")

    candles = load_structure_candles(
        config["filename"],
        config["target"],
    )

    print("Archive:", config["filename"])
    print("Candles Loaded:", len(candles))

    run_four_hour_event_rejection_audit(
        candles=candles,
        symbol=SYMBOL,
    )


if __name__ == "__main__":
    main()
