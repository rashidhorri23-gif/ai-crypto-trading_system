
import hashlib
import math
import re

from Src.Analysis.daily_structure import (
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)


VALID_DIRECTIONS = ("LONG", "SHORT")

VALID_EVENTS = {
    "BOS_BULLISH",
    "CHOCH_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BEARISH",
}


def _is_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_integer(value):
    return (
        not isinstance(value, bool)
        and _is_number(value)
        and int(value) == value
    )


def _empty_result(direction, reason, diagnostics=None):
    return {
        "status": "NO_SETUP",
        "allowed_direction": direction,
        "reason": reason,
        "setup": None,
        "diagnostics": diagnostics or {},
    }


def _reject(diagnostics, event, reason):
    diagnostics["rejected_candidate_count"] += 1

    diagnostics["rejected_candidates"].append({
        "event": event.get("event"),
        "timestamp": event.get("timestamp"),
        "event_index": event.get("index"),
        "reason": reason,
    })


def _zone_is_touched(candles, zone, start_index, end_index):
    for index in range(start_index, end_index + 1):
        candle = candles[index]

        if (
            candle["high"] >= zone["low"]
            and candle["low"] <= zone["high"]
        ):
            return True

    return False


def _find_order_blocks(
    candles,
    event_index,
    direction,
    lookback,
    broken_swing_index,
    diagnostics=None,
):
    if diagnostics is None:
        diagnostics = {}

    start = max(
        broken_swing_index + 1,
        event_index - lookback,
        0,
    )

    candidates = []

    for index in range(event_index - 1, start - 1, -1):
        diagnostics["ob_candidates_examined"] = (
            diagnostics.get("ob_candidates_examined", 0) + 1
        )

        candle = candles[index]

        if direction == "LONG":
            if candle["close"] >= candle["open"]:
                diagnostics["ob_wrong_candle_direction"] = (
                    diagnostics.get("ob_wrong_candle_direction", 0) + 1
                )
                continue

            zone = {
                "low": float(candle["low"]),
                "high": float(candle["open"]),
            }

        else:
            if candle["close"] <= candle["open"]:
                diagnostics["ob_wrong_candle_direction"] = (
                    diagnostics.get("ob_wrong_candle_direction", 0) + 1
                )
                continue

            zone = {
                "low": float(candle["open"]),
                "high": float(candle["high"]),
            }

        if zone["high"] <= zone["low"]:
            diagnostics["ob_invalid_zone"] = (
                diagnostics.get("ob_invalid_zone", 0) + 1
            )
            continue

        if _zone_is_touched(
            candles,
            zone,
            index + 1,
            event_index - 1,
        ):
            diagnostics["ob_mitigated"] = (
                diagnostics.get("ob_mitigated", 0) + 1
            )
            continue

        candidates.append({
            "index": index,
            "timestamp": candle["timestamp"],
            "low": zone["low"],
            "high": zone["high"],
            "direction": direction,
            "source": "ORDER_BLOCK",
            "structural_origin_index": broken_swing_index,
            "linked_event_index": event_index,
        })

    diagnostics["ob_valid_candidates"] = (
        diagnostics.get("ob_valid_candidates", 0) + len(candidates)
    )

    return candidates


def _find_order_block(
    candles,
    event_index,
    direction,
    lookback,
    broken_swing_index,
):
    candidates = _find_order_blocks(
        candles,
        event_index,
        direction,
        lookback,
        broken_swing_index,
    )

    return candidates[0] if candidates else None


def _find_fvgs(
    candles,
    event_index,
    direction,
    lookback,
    order_block,
    diagnostics=None,
):
    if order_block is None:
        return []

    if diagnostics is None:
        diagnostics = {}

    start = max(
        order_block["index"] + 2,
        event_index - lookback,
        2,
    )

    if start > event_index:
        diagnostics["fvg_search_windows_empty"] = (
            diagnostics.get("fvg_search_windows_empty", 0) + 1
        )
        return []

    candidates = []

    for index in range(event_index, start - 1, -1):
        diagnostics["fvg_patterns_checked"] = (
            diagnostics.get("fvg_patterns_checked", 0) + 1
        )

        first = candles[index - 2]
        middle = candles[index - 1]
        third = candles[index]

        if direction == "LONG":
            gap_valid = third["low"] > first["high"]

            if not gap_valid:
                diagnostics["fvg_no_gap"] = (
                    diagnostics.get("fvg_no_gap", 0) + 1
                )
                continue

            diagnostics["fvg_gap_condition_passed"] = (
                diagnostics.get("fvg_gap_condition_passed", 0) + 1
            )

            if middle["close"] <= middle["open"]:
                diagnostics["fvg_wrong_middle_direction"] = (
                    diagnostics.get("fvg_wrong_middle_direction", 0) + 1
                )
                continue

            diagnostics["fvg_middle_direction_passed"] = (
                diagnostics.get("fvg_middle_direction_passed", 0) + 1
            )

            zone_low = first["high"]
            zone_high = third["low"]
            source = "BULLISH_FVG"

        else:
            gap_valid = third["high"] < first["low"]

            if not gap_valid:
                diagnostics["fvg_no_gap"] = (
                    diagnostics.get("fvg_no_gap", 0) + 1
                )
                continue

            diagnostics["fvg_gap_condition_passed"] = (
                diagnostics.get("fvg_gap_condition_passed", 0) + 1
            )

            if middle["close"] >= middle["open"]:
                diagnostics["fvg_wrong_middle_direction"] = (
                    diagnostics.get("fvg_wrong_middle_direction", 0) + 1
                )
                continue

            diagnostics["fvg_middle_direction_passed"] = (
                diagnostics.get("fvg_middle_direction_passed", 0) + 1
            )

            zone_low = third["high"]
            zone_high = first["low"]
            source = "BEARISH_FVG"

        if zone_high <= zone_low:
            diagnostics["fvg_invalid_zone"] = (
                diagnostics.get("fvg_invalid_zone", 0) + 1
            )
            continue

        diagnostics["fvg_valid_patterns"] = (
            diagnostics.get("fvg_valid_patterns", 0) + 1
        )

        candidates.append({
            "index": index,
            "timestamp": third["timestamp"],
            "low": float(zone_low),
            "high": float(zone_high),
            "direction": direction,
            "source": source,
            "structural_origin_index": order_block[
                "structural_origin_index"
            ],
            "linked_event_index": event_index,
        })

    return candidates


def _find_fvg(
    candles,
    event_index,
    direction,
    lookback,
    order_block,
    diagnostics=None,
):
    candidates = _find_fvgs(
        candles,
        event_index,
        direction,
        lookback,
        order_block,
        diagnostics,
    )

    return candidates[0] if candidates else None


def _validate_zone_sequence(order_block, fvg):
    if order_block is None or fvg is None:
        return {
            "valid": False,
            "reason": "Both Order Block and FVG are required.",
        }

    if order_block["index"] >= fvg["index"]:
        return {
            "valid": False,
            "reason": "FVG must form after the Order Block.",
        }

    if fvg["index"] > order_block["linked_event_index"]:
        return {
            "valid": False,
            "reason": "FVG formed after the structure event.",
        }

    if (
        order_block["structural_origin_index"]
        != fvg["structural_origin_index"]
        or order_block["linked_event_index"]
        != fvg["linked_event_index"]
    ):
        return {
            "valid": False,
            "reason": "Zones do not share the same structural event.",
        }

    if order_block["direction"] != fvg["direction"]:
        return {
            "valid": False,
            "reason": "Zone directions do not match.",
        }

    return {
        "valid": True,
        "reason": "OB and FVG passed sequence validation.",
    }


def _calculate_overlap(order_block, fvg):
    if order_block is None or fvg is None:
        return {
            "confirmed": False,
            "low": None,
            "high": None,
            "reason": "Both zones are required.",
        }

    low = max(order_block["low"], fvg["low"])
    high = min(order_block["high"], fvg["high"])

    if high <= low:
        return {
            "confirmed": False,
            "low": None,
            "high": None,
            "reason": "Order Block and FVG do not overlap.",
        }

    return {
        "confirmed": True,
        "low": float(low),
        "high": float(high),
        "reason": "Order Block and FVG overlap.",
    }


def _swing_is_valid(swing, candles, event_index, direction):
    confirmation_index = swing.get("confirmation_index")
    swing_index = swing.get("index")
    price = swing.get("price")

    if (
        not _is_integer(confirmation_index)
        or not _is_integer(swing_index)
        or not _is_number(price)
    ):
        return False

    confirmation_index = int(confirmation_index)
    swing_index = int(swing_index)

    if (
        swing_index < 0
        or confirmation_index <= swing_index
        or confirmation_index >= event_index
        or price <= 0
    ):
        return False

    for index in range(confirmation_index + 1, event_index + 1):
        candle = candles[index]

        if direction == "LONG" and candle["low"] <= price:
            return False

        if direction == "SHORT" and candle["high"] >= price:
            return False

    return True


def _find_stop_swing(
    swings,
    candles,
    event_index,
    direction,
    entry_zone,
):
    required_type = "LOW" if direction == "LONG" else "HIGH"
    candidates = []

    for swing in swings:
        if swing.get("type") != required_type:
            continue

        price = swing.get("price")

        if not _is_number(price):
            continue

        if direction == "LONG" and price >= entry_zone["low"]:
            continue

        if direction == "SHORT" and price <= entry_zone["high"]:
            continue

        if not _swing_is_valid(
            swing,
            candles,
            event_index,
            direction,
        ):
            continue

        candidates.append(swing)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["confirmation_index"],
    )


def _build_setup_id(symbol, direction, event, order_block, fvg):
    raw = "|".join([
        symbol,
        direction,
        event["event"],
        str(event["timestamp"]),
        str(event.get("level")),
        str(order_block["index"]),
        str(fvg["index"]),
    ])

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return "{}-4H-{}-{}".format(
        symbol,
        direction,
        digest,
    )


def generate_four_hour_setup(
    candles,
    allowed_direction,
    setup_expiry_candles=24,
    ob_lookback=10,
    fvg_lookback=10,
    risk_reward=2.0,
    symbol="BTC-USDT",
    target_event=None,
):
    diagnostics = {
        "input_candles": 0,
        "candle_duration": 0,
        "swing_count": 0,
        "structure_events": 0,
        "matching_events": 0,
        "order_block_search_attempts": 0,
        "ob_candidates_examined": 0,
        "ob_wrong_candle_direction": 0,
        "ob_invalid_zone": 0,
        "ob_mitigated": 0,
        "ob_valid_candidates": 0,
        "order_blocks_found": 0,
        "fvg_search_windows_empty": 0,
        "fvg_patterns_checked": 0,
        "fvg_no_gap": 0,
        "fvg_gap_condition_passed": 0,
        "fvg_wrong_middle_direction": 0,
        "fvg_middle_direction_passed": 0,
        "fvg_invalid_zone": 0,
        "fvg_valid_patterns": 0,
        "fvg_found": 0,
        "ob_fvg_overlap_confirmed": 0,
        "ob_fvg_overlap_failed": 0,
        "valid_stop_swings": 0,
        "evaluated_candidate_count": 0,
        "rejected_candidate_count": 0,
        "rejected_candidates": [],
        "evaluated_candidates": [],
        "selected_setup_id": None,
    }

    if allowed_direction not in VALID_DIRECTIONS:
        return _empty_result(
            allowed_direction,
            "Allowed direction must be LONG or SHORT.",
            diagnostics,
        )

    if not isinstance(symbol, str) or not symbol.strip():
        return _empty_result(
            allowed_direction,
            "Invalid symbol.",
            diagnostics,
        )

    symbol = re.sub(
        r"[^A-Z0-9_-]",
        "",
        symbol.strip().upper(),
    )

    if not symbol:
        return _empty_result(
            allowed_direction,
            "Invalid symbol.",
            diagnostics,
        )

    for name, value in (
        ("setup_expiry_candles", setup_expiry_candles),
        ("ob_lookback", ob_lookback),
        ("fvg_lookback", fvg_lookback),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            return _empty_result(
                allowed_direction,
                "{} must be a positive integer.".format(name),
                diagnostics,
            )

    if not _is_number(risk_reward) or risk_reward <= 0:
        return _empty_result(
            allowed_direction,
            "Invalid risk reward.",
            diagnostics,
        )

    if target_event is not None:
        if not isinstance(target_event, dict):
            return _empty_result(
                allowed_direction,
                "target_event must be a dictionary.",
                diagnostics,
            )

        target_index = target_event.get("index")
        target_timestamp = target_event.get("timestamp")
        target_name = target_event.get("event")

        if (
            not _is_integer(target_index)
            or not _is_integer(target_timestamp)
            or target_name not in VALID_EVENTS
        ):
            return _empty_result(
                allowed_direction,
                "Invalid target event fields.",
                diagnostics,
            )

        expected_target_direction = (
            "BULLISH" if allowed_direction == "LONG" else "BEARISH"
        )

        if not target_name.endswith(expected_target_direction):
            return _empty_result(
                allowed_direction,
                "Target event direction does not match allowed direction.",
                diagnostics,
            )

        diagnostics["target_event_filter"] = {
            "event": target_name,
            "index": int(target_index),
            "timestamp": int(target_timestamp),
        }

    try:
        candles = validate_candles(list(candles))
        duration = infer_candle_duration(candles)

        if len(candles) < 20:
            raise ValueError("At least 20 candles are required.")

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

    except Exception as error:
        return _empty_result(
            allowed_direction,
            "Analysis failed: {}: {}".format(
                type(error).__name__,
                error,
            ),
            diagnostics,
        )

    diagnostics["input_candles"] = len(candles)
    diagnostics["candle_duration"] = duration
    diagnostics["swing_count"] = len(swings)
    diagnostics["structure_events"] = len(events)

    current_close = candles[-1]["timestamp"] + duration

    required_direction = (
        "BULLISH" if allowed_direction == "LONG" else "BEARISH"
    )

    matching = [
        event for event in events
        if (
            event.get("direction") == required_direction
            and event.get("event") in VALID_EVENTS
            and _is_integer(event.get("available_timestamp"))
            and event["available_timestamp"] <= current_close
        )
    ]

    if target_event is not None:
        matching = [
            event for event in matching
            if (
                event.get("event") == target_event["event"]
                and event.get("index") == target_event["index"]
                and event.get("timestamp") == target_event["timestamp"]
            )
        ]

    matching.sort(
        key=lambda item: (
            item["available_timestamp"],
            item["index"],
        ),
        reverse=True,
    )

    diagnostics["matching_events"] = len(matching)

    if not matching:
        return _empty_result(
            allowed_direction,
            "The requested confirmed structure event was not reproduced."
            if target_event is not None
            else "No confirmed event exists for this direction.",
            diagnostics,
        )

    for event in matching:
        event_index = event.get("index")
        broken_index = event.get("broken_swing_index")

        if (
            not _is_integer(event_index)
            or not _is_integer(broken_index)
        ):
            _reject(diagnostics, event, "Invalid event indices.")
            continue

        event_index = int(event_index)
        broken_index = int(broken_index)

        if (
            event_index <= broken_index
            or event_index >= len(candles)
        ):
            _reject(diagnostics, event, "Invalid structural origin.")
            continue

        expected_type = (
            "HIGH" if allowed_direction == "LONG" else "LOW"
        )

        if event.get("broken_swing_type") != expected_type:
            _reject(diagnostics, event, "Broken swing type mismatch.")
            continue

        if event.get("timestamp") != candles[event_index]["timestamp"]:
            _reject(diagnostics, event, "Event timestamp mismatch.")
            continue

        available = event.get("available_timestamp")

        if (
            not _is_integer(available)
            or available != event["timestamp"] + duration
        ):
            _reject(diagnostics, event, "Invalid event availability time.")
            continue

        expiry = int(available) + setup_expiry_candles * duration

        if current_close >= expiry:
            _reject(diagnostics, event, "Setup has expired.")
            continue

        diagnostics["evaluated_candidate_count"] += 1

        diagnostics["evaluated_candidates"].append({
            "event": event["event"],
            "event_index": event_index,
            "broken_swing_index": broken_index,
            "timestamp": event["timestamp"],
        })

        diagnostics["order_block_search_attempts"] += 1

        order_blocks = _find_order_blocks(
            candles,
            event_index,
            allowed_direction,
            ob_lookback,
            broken_index,
            diagnostics,
        )

        diagnostics["order_blocks_found"] += len(order_blocks)

        if not order_blocks:
            _reject(
                diagnostics,
                event,
                "No valid unmitigated Order Block.",
            )
            continue

        any_fvg = False
        any_overlap = False
        any_stop_swing = False
        final_rejection = "No valid OB/FVG combination."

        for ob in order_blocks:
            fvgs = _find_fvgs(
                candles,
                event_index,
                allowed_direction,
                fvg_lookback,
                ob,
                diagnostics,
            )

            diagnostics["fvg_found"] += len(fvgs)

            if fvgs:
                any_fvg = True

            for fvg in fvgs:
                sequence = _validate_zone_sequence(ob, fvg)

                if not sequence["valid"]:
                    final_rejection = sequence["reason"]
                    continue

                overlap = _calculate_overlap(ob, fvg)

                if not overlap["confirmed"]:
                    diagnostics["ob_fvg_overlap_failed"] += 1
                    final_rejection = overlap["reason"]
                    continue

                diagnostics["ob_fvg_overlap_confirmed"] += 1
                any_overlap = True

                entry_zone = {
                    "source": "OB_FVG_OVERLAP",
                    "low": overlap["low"],
                    "high": overlap["high"],
                }

                entry_reference = (
                    entry_zone["low"] + entry_zone["high"]
                ) / 2.0

                stop_swing = _find_stop_swing(
                    swings,
                    candles,
                    event_index,
                    allowed_direction,
                    entry_zone,
                )

                if stop_swing is None:
                    final_rejection = "No valid opposite stop swing."
                    continue

                any_stop_swing = True
                diagnostics["valid_stop_swings"] += 1

                stop_loss = float(stop_swing["price"])

                if allowed_direction == "LONG":
                    risk_distance = entry_reference - stop_loss
                    take_profit = (
                        entry_reference + risk_distance * risk_reward
                    )
                else:
                    risk_distance = stop_loss - entry_reference
                    take_profit = (
                        entry_reference - risk_distance * risk_reward
                    )

                if risk_distance <= 0 or take_profit <= 0:
                    final_rejection = "Invalid risk or target."
                    continue

                setup_id = _build_setup_id(
                    symbol,
                    allowed_direction,
                    event,
                    ob,
                    fvg,
                )

                setup = {
                    "setup_id": setup_id,
                    "symbol": symbol,
                    "timeframe": "4H",
                    "direction": allowed_direction,
                    "status": "CONFIRMED_CONFLUENCE",
                    "structure_event": event["event"],
                    "structure_event_timestamp": int(event["timestamp"]),
                    "available_timestamp": int(available),
                    "expiry_timestamp": expiry,
                    "entry_zone": entry_zone,
                    "entry_reference": float(entry_reference),
                    "order_block": ob,
                    "fvg": fvg,
                    "zone_overlap": overlap,
                    "zone_sequence": sequence,
                    "stop_swing": {
                        "type": stop_swing["type"],
                        "index": int(stop_swing["index"]),
                        "timestamp": stop_swing["timestamp"],
                        "confirmation_index": int(
                            stop_swing["confirmation_index"]
                        ),
                        "confirmation_timestamp": stop_swing.get(
                            "confirmation_timestamp"
                        ),
                        "price": stop_loss,
                    },
                    "stop_loss": stop_loss,
                    "take_profit": float(take_profit),
                    "risk_distance": float(risk_distance),
                    "risk_reward": float(risk_reward),
                }

                diagnostics["selected_setup_id"] = setup_id
                diagnostics["selected_event"] = event["event"]
                diagnostics["selected_event_index"] = event_index

                return {
                    "status": "CONFIRMED_CONFLUENCE",
                    "allowed_direction": allowed_direction,
                    "reason": (
                        "A valid 4H event, unmitigated Order Block, "
                        "subsequent FVG, overlap and opposite stop swing "
                        "were confirmed."
                    ),
                    "setup": setup,
                    "diagnostics": diagnostics,
                }

        if not any_fvg:
            final_rejection = "No valid FVG after any valid Order Block."
        elif not any_overlap:
            final_rejection = "No overlapping OB/FVG pair."
        elif not any_stop_swing:
            final_rejection = "No valid opposite stop swing."

        _reject(diagnostics, event, final_rejection)

    return _empty_result(
        allowed_direction,
        "No candidate passed all 4H setup validation rules.",
        diagnostics,
    )
