
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

VALID_EVENTS = (
    "BOS_BULLISH",
    "CHOCH_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BEARISH",
)


def _is_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_integer(value):
    if isinstance(value, bool):
        return False

    try:
        return int(value) == value
    except (TypeError, ValueError, OverflowError):
        return False


def _empty_result(direction, reason, diagnostics=None):
    return {
        "status": "NO_SETUP",
        "allowed_direction": direction,
        "reason": reason,
        "setup": None,
        "diagnostics": diagnostics or {},
    }


def _rejection(diagnostics, event, reason):
    diagnostics["rejected_candidates"].append({
        "event": event.get("event"),
        "timestamp": event.get("timestamp"),
        "event_index": event.get("index"),
        "reason": reason,
    })


def _find_order_block(
    candles,
    event_index,
    direction,
    lookback,
    broken_swing_index,
):
    start_index = max(
        0,
        event_index - lookback,
        broken_swing_index + 1,
    )

    for index in range(event_index - 1, start_index - 1, -1):
        candle = candles[index]

        if direction == "LONG":
            if candle["close"] >= candle["open"]:
                continue

            zone_low = candle["low"]
            zone_high = candle["open"]

        else:
            if candle["close"] <= candle["open"]:
                continue

            zone_low = candle["open"]
            zone_high = candle["high"]

        if (
            not _is_number(zone_low)
            or not _is_number(zone_high)
            or zone_low <= 0
            or zone_high <= zone_low
        ):
            continue

        return {
            "index": index,
            "timestamp": candle["timestamp"],
            "low": float(zone_low),
            "high": float(zone_high),
            "direction": direction,
            "source": "ORDER_BLOCK",
            "structural_origin_index": broken_swing_index,
            "linked_event_index": event_index,
        }

    return None


def _find_fvg(
    candles,
    event_index,
    direction,
    lookback,
    broken_swing_index,
):
    start_index = max(
        2,
        event_index - lookback,
        broken_swing_index + 2,
    )

    for index in range(event_index, start_index - 1, -1):
        first = candles[index - 2]
        third = candles[index]

        if direction == "LONG":
            if third["low"] > first["high"]:
                return {
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": float(first["high"]),
                    "high": float(third["low"]),
                    "direction": direction,
                    "source": "BULLISH_FVG",
                    "structural_origin_index": broken_swing_index,
                    "linked_event_index": event_index,
                }

        else:
            if third["high"] < first["low"]:
                return {
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": float(third["high"]),
                    "high": float(first["low"]),
                    "direction": direction,
                    "source": "BEARISH_FVG",
                    "structural_origin_index": broken_swing_index,
                    "linked_event_index": event_index,
                }

    return None


def _validate_zone_sequence(order_block, fvg):
    if order_block is None or fvg is None:
        return {
            "valid": True,
            "reason": "Only one zone was found; confluence is not confirmed.",
        }

    if order_block["index"] >= fvg["index"]:
        return {
            "valid": False,
            "reason": "FVG does not follow the selected Order Block.",
        }

    if order_block["linked_event_index"] != fvg["linked_event_index"]:
        return {
            "valid": False,
            "reason": "Order Block and FVG are not linked to the same event.",
        }

    if (
        order_block["structural_origin_index"]
        != fvg["structural_origin_index"]
    ):
        return {
            "valid": False,
            "reason": "Order Block and FVG have different structural origins.",
        }

    return {
        "valid": True,
        "reason": "Order Block and FVG belong to the same structural leg.",
    }


def _calculate_overlap(order_block, fvg):
    if order_block is None or fvg is None:
        return {
            "confirmed": False,
            "low": None,
            "high": None,
            "reason": "Both zones are required for confluence.",
        }

    overlap_low = max(
        order_block["low"],
        fvg["low"],
    )

    overlap_high = min(
        order_block["high"],
        fvg["high"],
    )

    if overlap_high <= overlap_low:
        return {
            "confirmed": False,
            "low": None,
            "high": None,
            "reason": "Order Block and FVG do not overlap.",
        }

    return {
        "confirmed": True,
        "low": float(overlap_low),
        "high": float(overlap_high),
        "reason": "Order Block and FVG overlap.",
    }


def _swing_is_valid(
    swing,
    candles,
    event_index,
    direction,
):
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

    if confirmation_index >= event_index:
        return False

    if swing_index >= confirmation_index:
        return False

    if price <= 0:
        return False

    for index in range(confirmation_index + 1, event_index + 1):
        candle = candles[index]

        if direction == "LONG":
            if candle["low"] <= price:
                return False

        else:
            if candle["high"] >= price:
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
        swing_type = str(swing.get("type", "")).upper()

        if swing_type not in (required_type, "SWING_" + required_type):
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
        key=lambda item: int(item["confirmation_index"]),
    )


def _build_setup_id(
    symbol,
    direction,
    event,
    order_block,
    fvg,
):
    raw_id = "|".join([
        symbol,
        direction,
        str(event["event"]),
        str(event["timestamp"]),
        str(event.get("level")),
        str(order_block["index"]) if order_block else "NO_OB",
        str(fvg["index"]) if fvg else "NO_FVG",
    ])

    digest = hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()[:16]

    return f"{symbol}-4H-{direction}-{digest}"


def _build_entry_zone(order_block, fvg, overlap):
    if overlap["confirmed"]:
        return {
            "source": "OB_FVG_OVERLAP",
            "low": overlap["low"],
            "high": overlap["high"],
        }

    if order_block is not None:
        return {
            "source": "ORDER_BLOCK",
            "low": order_block["low"],
            "high": order_block["high"],
        }

    if fvg is not None:
        return {
            "source": "FVG",
            "low": fvg["low"],
            "high": fvg["high"],
        }

    return None


def generate_four_hour_setup(
    candles,
    allowed_direction,
    setup_expiry_candles=24,
    ob_lookback=10,
    fvg_lookback=10,
    risk_reward=2.0,
    symbol="BTC-USDT",
):
    diagnostics = {
        "input_candles": 0,
        "structure_events": 0,
        "matching_events": 0,
        "order_blocks_found": 0,
        "fvg_found": 0,
        "valid_stop_swings": 0,
        "rejected_candidates": [],
        "evaluated_candidates": [],
        "selected_setup_id": None,
        "selected_event": None,
        "selected_event_index": None,
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
            "Symbol must be a non-empty string.",
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
            "Symbol contains no valid characters.",
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
                f"{name} must be a positive integer.",
                diagnostics,
            )

    if not _is_number(risk_reward) or risk_reward <= 0:
        return _empty_result(
            allowed_direction,
            "risk_reward must be a positive number.",
            diagnostics,
        )

    try:
        validated = validate_candles(list(candles))
        candle_duration = infer_candle_duration(validated)

    except Exception as error:
        return _empty_result(
            allowed_direction,
            f"Invalid candle data: {type(error).__name__}: {error}",
            diagnostics,
        )

    diagnostics["input_candles"] = len(validated)
    diagnostics["candle_duration"] = candle_duration

    if len(validated) < 20:
        return _empty_result(
            allowed_direction,
            "At least 20 candles are required.",
            diagnostics,
        )

    if (
        not _is_integer(candle_duration)
        or candle_duration <= 0
    ):
        return _empty_result(
            allowed_direction,
            "Invalid candle duration.",
            diagnostics,
        )

    current_candle = validated[-1]

    current_close_timestamp = (
        current_candle["timestamp"] + candle_duration
    )

    try:
        swings = detect_swings(
            validated,
            left_bars=3,
            right_bars=3,
            candle_duration=candle_duration,
        )

        events = analyze_structure_events(
            validated,
            swings,
            candle_duration,
        )

    except Exception as error:
        return _empty_result(
            allowed_direction,
            f"Structure analysis failed: {type(error).__name__}: {error}",
            diagnostics,
        )

    diagnostics["swing_count"] = len(swings)
    diagnostics["structure_events"] = len(events)

    required_event_direction = (
        "BULLISH" if allowed_direction == "LONG" else "BEARISH"
    )

    matching_events = []

    for event in events:
        if event.get("direction") != required_event_direction:
            continue

        if event.get("event") not in VALID_EVENTS:
            continue

        available_timestamp = event.get("available_timestamp")

        if not _is_integer(available_timestamp):
            continue

        if available_timestamp > current_close_timestamp:
            continue

        matching_events.append(event)

    matching_events.sort(
        key=lambda event: (
            int(event["available_timestamp"]),
            int(event.get("index", -1)),
        ),
        reverse=True,
    )

    diagnostics["matching_events"] = len(matching_events)

    if not matching_events:
        return _empty_result(
            allowed_direction,
            "No confirmed structure event exists for this direction.",
            diagnostics,
        )

    selected_candidate = None

    for event in matching_events:
        event_index = event.get("index")
        broken_swing_index = event.get("broken_swing_index")

        if (
            not _is_integer(event_index)
            or not _is_integer(broken_swing_index)
        ):
            _rejection(
                diagnostics,
                event,
                "Missing or invalid structural origin index.",
            )
            continue

        event_index = int(event_index)
        broken_swing_index = int(broken_swing_index)

        if (
            event_index < 0
            or event_index >= len(validated)
            or broken_swing_index < 0
            or broken_swing_index >= event_index
        ):
            _rejection(
                diagnostics,
                event,
                "Invalid structural origin or event index.",
            )
            continue

        expected_broken_type = (
            "HIGH" if allowed_direction == "LONG" else "LOW"
        )

        actual_broken_type = str(
            event.get("broken_swing_type", "")
        ).upper()

        if actual_broken_type not in (
            expected_broken_type,
            "SWING_" + expected_broken_type,
        ):
            _rejection(
                diagnostics,
                event,
                "Broken swing type does not match event direction.",
            )
            continue

        event_timestamp = event.get("timestamp")
        available_timestamp = event.get("available_timestamp")

        if (
            not _is_integer(event_timestamp)
            or not _is_integer(available_timestamp)
        ):
            _rejection(
                diagnostics,
                event,
                "Invalid event or availability timestamp.",
            )
            continue

        if (
            event_timestamp != validated[event_index]["timestamp"]
            or available_timestamp < event_timestamp
        ):
            _rejection(
                diagnostics,
                event,
                "Event timestamps do not match the source candle.",
            )
            continue

        diagnostics["evaluated_candidates"].append({
            "event": event["event"],
            "event_index": event_index,
            "broken_swing_index": broken_swing_index,
            "timestamp": event_timestamp,
        })

        order_block = _find_order_block(
            validated,
            event_index,
            allowed_direction,
            ob_lookback,
            broken_swing_index,
        )

        fvg = _find_fvg(
            validated,
            event_index,
            allowed_direction,
            fvg_lookback,
            broken_swing_index,
        )

        if order_block is not None:
            diagnostics["order_blocks_found"] += 1

        if fvg is not None:
            diagnostics["fvg_found"] += 1

        if order_block is None and fvg is None:
            _rejection(
                diagnostics,
                event,
                "No structurally linked Order Block or FVG found.",
            )
            continue

        sequence = _validate_zone_sequence(
            order_block,
            fvg,
        )

        if not sequence["valid"]:
            _rejection(
                diagnostics,
                event,
                sequence["reason"],
            )
            continue

        overlap = _calculate_overlap(
            order_block,
            fvg,
        )

        entry_zone = _build_entry_zone(
            order_block,
            fvg,
            overlap,
        )

        if entry_zone is None:
            _rejection(
                diagnostics,
                event,
                "No valid entry zone found.",
            )
            continue

        entry_reference = (
            entry_zone["low"] + entry_zone["high"]
        ) / 2.0

        stop_swing = _find_stop_swing(
            swings,
            validated,
            event_index,
            allowed_direction,
            entry_zone,
        )

        if stop_swing is None:
            _rejection(
                diagnostics,
                event,
                "No confirmed and unbroken opposite swing exists beyond the entry zone.",
            )
            continue

        diagnostics["valid_stop_swings"] += 1

        stop_loss = float(stop_swing["price"])

        if allowed_direction == "LONG":
            risk_distance = entry_reference - stop_loss
        else:
            risk_distance = stop_loss - entry_reference

        if not _is_number(risk_distance) or risk_distance <= 0:
            _rejection(
                diagnostics,
                event,
                "Invalid risk distance.",
            )
            continue

        if allowed_direction == "LONG":
            if stop_loss >= entry_zone["low"]:
                _rejection(
                    diagnostics,
                    event,
                    "LONG stop loss must be below the complete entry zone.",
                )
                continue

            take_profit = (
                entry_reference + risk_distance * risk_reward
            )

        else:
            if stop_loss <= entry_zone["high"]:
                _rejection(
                    diagnostics,
                    event,
                    "SHORT stop loss must be above the complete entry zone.",
                )
                continue

            take_profit = (
                entry_reference - risk_distance * risk_reward
            )

        if take_profit <= 0:
            _rejection(
                diagnostics,
                event,
                "Calculated take profit is invalid.",
            )
            continue

        expiry_timestamp = (
            int(available_timestamp)
            + setup_expiry_candles * candle_duration
        )

        if current_close_timestamp > expiry_timestamp:
            setup_status = "EXPIRED"
        elif overlap["confirmed"]:
            setup_status = "CONFIRMED_CONFLUENCE"
        else:
            setup_status = "WATCH_NO_ZONE_OVERLAP"

        setup_id = _build_setup_id(
            symbol,
            allowed_direction,
            event,
            order_block,
            fvg,
        )

        selected_candidate = {
            "setup_id": setup_id,
            "symbol": symbol,
            "timeframe": "4H",
            "direction": allowed_direction,
            "status": setup_status,
            "structure_event": event["event"],
            "structure_event_timestamp": int(event_timestamp),
            "available_timestamp": int(available_timestamp),
            "expiry_timestamp": expiry_timestamp,
            "entry_zone": entry_zone,
            "entry_reference": float(entry_reference),
            "order_block": order_block,
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

        break

    if selected_candidate is None:
        return _empty_result(
            allowed_direction,
            "No valid setup passed the structural and risk checks.",
            diagnostics,
        )

    if selected_candidate["status"] == "EXPIRED":
        result_status = "EXPIRED"
        reason = "The latest valid setup has exceeded its expiry time."

    elif selected_candidate["status"] == "CONFIRMED_CONFLUENCE":
        result_status = "CONFIRMED_CONFLUENCE"
        reason = (
            "Confirmed structure event with structurally linked "
            "and overlapping Order Block and FVG."
        )

    else:
        result_status = "WATCH_NO_ZONE_OVERLAP"
        reason = (
            "A structurally valid setup exists, but Order Block "
            "and FVG do not overlap."
        )

    return {
        "status": result_status,
        "allowed_direction": allowed_direction,
        "reason": reason,
        "setup": selected_candidate,
        "diagnostics": diagnostics,
    }
