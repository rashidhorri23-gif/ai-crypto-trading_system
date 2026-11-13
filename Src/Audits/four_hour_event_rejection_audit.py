import csv
import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from Src.Main import TIMEFRAME_CONFIG, load_structure_candles, SYMBOL

from Src.Analysis.daily_structure import (
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)

from Src.Analysis.four_hour_setup import generate_four_hour_setup


DIRECTIONS = {
    "LONG": "BULLISH",
    "SHORT": "BEARISH",
}

MITIGATION_MODES = ("A", "B", "C")

SETUP_EXPIRY_CANDLES = 24
OB_LOOKBACK = 10
FVG_LOOKBACK = 10
RISK_REWARD = 2.0

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "Reports"
    / "4H_Event_Rejection_Audit"
)

SUMMARY_FILE = OUTPUT_DIR / "4h_event_rejection_summary.csv"
DETAILS_FILE = OUTPUT_DIR / "4h_event_rejection_details.jsonl"

NUMERIC_DIAGNOSTIC_KEYS = (
    "input_candles",
    "swing_count",
    "structure_events",
    "matching_events",
    "order_block_search_attempts",
    "ob_candidates_examined",
    "ob_wrong_candle_direction",
    "ob_invalid_zone",
    "ob_mitigated",
    "ob_mitigated_before_or_during_fvg",
    "ob_mitigated_after_fvg",
    "ob_unmitigated_in_window",
    "ob_unmitigated_from_ob",
    "ob_unmitigated_after_fvg",
    "ob_valid_candidates",
    "order_blocks_found",
    "fvg_search_windows_empty",
    "fvg_patterns_checked",
    "fvg_no_gap",
    "fvg_gap_condition_passed",
    "fvg_wrong_middle_direction",
    "fvg_middle_direction_passed",
    "fvg_invalid_zone",
    "fvg_valid_patterns",
    "fvg_found",
    "ob_fvg_pairs_checked",
    "ob_fvg_overlap_confirmed",
    "ob_fvg_overlap_failed",
    "valid_stop_swings",
    "evaluated_candidate_count",
    "expired_candidate_count",
    "rejected_candidate_count",
)


def write_json_line(file_handle, data):
    file_handle.write(
        json.dumps(
            data,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )


def safe_integer(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float) and math.isfinite(value):
        if int(value) == value:
            return int(value)

    return None


def create_summary_group():
    return {
        "event_count": 0,
        "confirmed_count": 0,
        "no_setup_count": 0,
        "error_count": 0,
        "result_reasons": Counter(),
        "pair_statuses": Counter(),
        "rejection_reasons": Counter(),
        "numeric_totals": defaultdict(int),
    }


def add_diagnostics(group, diagnostics):
    if not isinstance(diagnostics, dict):
        return

    for key in NUMERIC_DIAGNOSTIC_KEYS:
        value = diagnostics.get(key, 0)

        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            group["numeric_totals"][key] += value

    for item in diagnostics.get("ob_fvg_pair_details", []):
        if isinstance(item, dict):
            status = item.get("status", "UNKNOWN")
            group["pair_statuses"][str(status)] += 1

    for item in diagnostics.get("rejected_candidates", []):
        if isinstance(item, dict):
            reason = item.get("reason", "UNKNOWN")
            group["rejection_reasons"][str(reason)] += 1


def evaluate_event(candles, event, direction, mitigation_mode):
    event_index = safe_integer(event.get("index"))
    event_timestamp = safe_integer(event.get("timestamp"))
    event_name = event.get("event")

    if event_index is None or event_timestamp is None:
        raise ValueError("Event index or timestamp is invalid.")

    if event_index < 0 or event_index >= len(candles):
        raise ValueError("Event index is outside candle history.")

    if candles[event_index]["timestamp"] != event_timestamp:
        raise ValueError("Event timestamp does not match its candle.")

    # Evaluate the event at its own confirmation time.
    # This prevents later candles from influencing the result.
    historical_candles = candles[:event_index + 1]

    target_event = {
        "event": event_name,
        "index": event_index,
        "timestamp": event_timestamp,
    }

    return generate_four_hour_setup(
        candles=historical_candles,
        allowed_direction=direction,
        setup_expiry_candles=SETUP_EXPIRY_CANDLES,
        ob_lookback=OB_LOOKBACK,
        fvg_lookback=FVG_LOOKBACK,
        risk_reward=RISK_REWARD,
        symbol=SYMBOL,
        target_event=target_event,
        mitigation_mode=mitigation_mode,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timeframe_config = TIMEFRAME_CONFIG.get("4H")

    if not isinstance(timeframe_config, dict):
        raise RuntimeError(
            "The 4H timeframe configuration was not found in Src.Main."
        )

    filename = timeframe_config.get("filename")

    if not filename:
        raise RuntimeError(
            "The 4H archive filename is missing from TIMEFRAME_CONFIG."
        )

    print("=" * 72)
    print("4H EVENT REJECTION AUDIT")
    print("=" * 72)
    print("Symbol:", SYMBOL)
    print("Archive:", filename)
    print("Mitigation modes:", ", ".join(MITIGATION_MODES))
    print("Output directory:", OUTPUT_DIR)
    print()

    raw_candles = load_structure_candles(filename, 540)

    if not raw_candles:
        raise RuntimeError("No 4H candles were loaded.")

    candles = validate_candles(list(raw_candles))

    if len(candles) < 100:
        raise RuntimeError(
            "Insufficient 4H history. At least 100 candles are required."
        )

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

    print("Candles loaded:", len(candles))
    print("Candle duration:", duration)
    print("Swings detected:", len(swings))
    print("Structure events detected:", len(events))
    print()

    event_groups = {
        direction: [
            event
            for event in events
            if (
                event.get("direction") == expected_direction
                and event.get("event") in {
                    "BOS_BULLISH",
                    "CHOCH_BULLISH",
                    "BOS_BEARISH",
                    "CHOCH_BEARISH",
                }
            )
        ]
        for direction, expected_direction in DIRECTIONS.items()
    }

    summaries = {
        (direction, mode): create_summary_group()
        for direction in DIRECTIONS
        for mode in MITIGATION_MODES
    }

    total_runs = 0
    total_errors = 0

    with DETAILS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as details_handle:

        for direction, direction_events in event_groups.items():
            direction_events.sort(
                key=lambda item: (
                    safe_integer(item.get("available_timestamp")) or 0,
                    safe_integer(item.get("index")) or 0,
                )
            )

            print(
                "{} events: {}".format(
                    direction,
                    len(direction_events),
                )
            )

            for event_number, event in enumerate(
                direction_events,
                start=1,
            ):
                for mitigation_mode in MITIGATION_MODES:
                    group = summaries[(direction, mitigation_mode)]

                    total_runs += 1
                    group["event_count"] += 1

                    try:
                        result = evaluate_event(
                            candles=candles,
                            event=event,
                            direction=direction,
                            mitigation_mode=mitigation_mode,
                        )

                        diagnostics = result.get("diagnostics", {})
                        status = result.get("status", "UNKNOWN")
                        reason = result.get("reason", "No reason returned.")

                        if status == "CONFIRMED_CONFLUENCE":
                            group["confirmed_count"] += 1
                        else:
                            group["no_setup_count"] += 1

                        group["result_reasons"][str(reason)] += 1

                        add_diagnostics(
                            group,
                            diagnostics,
                        )

                        detail_record = {
                            "direction": direction,
                            "mitigation_mode": mitigation_mode,
                            "event_number": event_number,
                            "event": event,
                            "status": status,
                            "reason": reason,
                            "setup": result.get("setup"),
                            "diagnostics": diagnostics,
                            "error": None,
                        }

                    except Exception as error:
                        total_errors += 1
                        group["error_count"] += 1

                        detail_record = {
                            "direction": direction,
                            "mitigation_mode": mitigation_mode,
                            "event_number": event_number,
                            "event": event,
                            "status": "ERROR",
                            "reason": str(error),
                            "setup": None,
                            "diagnostics": {},
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        }

                    write_json_line(
                        details_handle,
                        detail_record,
                    )

                    print(
                        "[{}/{}] {} {} | {} | {}".format(
                            event_number,
                            len(direction_events),
                            direction,
                            mitigation_mode,
                            detail_record["status"],
                            detail_record["reason"],
                        )
                    )

    summary_rows = []

    for direction in DIRECTIONS:
        for mitigation_mode in MITIGATION_MODES:
            group = summaries[(direction, mitigation_mode)]

            summary_rows.append({
                "direction": direction,
                "mitigation_mode": mitigation_mode,
                "events_tested": group["event_count"],
                "confirmed_setups": group["confirmed_count"],
                "no_setup_results": group["no_setup_count"],
                "errors": group["error_count"],
                "numeric_diagnostics": json.dumps(
                    dict(group["numeric_totals"]),
                    ensure_ascii=False,
                ),
                "pair_status_counts": json.dumps(
                    dict(group["pair_statuses"]),
                    ensure_ascii=False,
                ),
                "rejection_reason_counts": json.dumps(
                    dict(group["rejection_reasons"]),
                    ensure_ascii=False,
                ),
                "result_reason_counts": json.dumps(
                    dict(group["result_reasons"]),
                    ensure_ascii=False,
                ),
            })

    fieldnames = [
        "direction",
        "mitigation_mode",
        "events_tested",
        "confirmed_setups",
        "no_setup_results",
        "errors",
        "numeric_diagnostics",
        "pair_status_counts",
        "rejection_reason_counts",
        "result_reason_counts",
    ]

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as summary_handle:
        writer = csv.DictWriter(
            summary_handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 72)
    print("AUDIT COMPLETED")
    print("=" * 72)
    print("Total event evaluations:", total_runs)
    print("Errors:", total_errors)
    print("Summary:", SUMMARY_FILE)
    print("Details:", DETAILS_FILE)
    print()

    for row in summary_rows:
        print(
            "{} | Mode {} | Events: {} | Confirmed: {} | Errors: {}".format(
                row["direction"],
                row["mitigation_mode"],
                row["events_tested"],
                row["confirmed_setups"],
                row["errors"],
            )
        )

    if total_errors:
        print()
        print(
            "WARNING: Some evaluations failed. "
            "Inspect the JSONL details before interpreting the results."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
