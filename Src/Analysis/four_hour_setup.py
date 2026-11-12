
import hashlib
import math

from Src.Analysis.daily_structure import (
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)


def _is_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _empty_result(direction, reason, diagnostics=None):
    return {
        "status": "NO_SETUP",
        "allowed_direction": direction,
        "reason": reason,
        "setup": None,
        "diagnostics": diagnostics or {},
    }


def _find_order_block(candles, event_index, direction, lookback):
    start_index = max(0, event_index - lookback)

    for index in range(event_index - 1, start_index - 1, -1):
        candle = candles[index]

        if direction == "LONG":
            is_opposite = candle["close"] < candle["open"]

            if not is_opposite:
                continue

            zone_low = candle["low"]
            zone_high = candle["open"]

        else:
            is_opposite = candle["close"] > candle["open"]

            if not is_opposite:
                continue

            zone_low = candle["open"]
            zone_high = candle["high"]

        if zone_low <= 0 or zone_high <= zone_low:
            continue

        return {
            "index": index,
            "timestamp": candle["timestamp"],
            "low": zone_low,
            "high": zone_high,
            "direction": direction,
            "source": "ORDER_BLOCK",
        }

    return None


def _find_fvg(candles, event_index, direction, lookback):
    start_index = max(2, event_index - lookback)

    for index in range(event_index, start_index - 1, -1):
        first = candles[index - 2]
        third = candles[index]

        if direction == "LONG":
            if third["low"] > first["high"]:
                return {
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": first["high"],
                    "high": third["low"],
                    "direction": direction,
                    "source": "BULLISH_FVG",
                }

        else:
            if third["high"] < first["low"]:
                return {
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": third["high"],
                    "high": first["low"],
                    "direction": direction,
                    "source": "BEARISH_FVG",
                }

    return None


def _find_stop_swing(
    swings,
    event_index,
    direction,
    entry_reference,
):
    required_type = "LOW" if direction == "LONG" else "HIGH"

    candidates = []

    for swing in swings:
        if swing.get("type") != required_type:
            continue

        confirmation_index = swing.get("confirmation_index")
        price = swing.get("price")

        if (
            not isinstance(confirmation_index, int)
            or confirmation_index >= event_index
            or not _is_number(price)
        ):
            continue

        if direction == "LONG" and price >= entry_reference:
            continue

        if direction == "SHORT" and price <= entry_reference:
            continue

        candidates.append(swing)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["confirmation_index"],
    )


def _calculate_overlap(order_block, fvg):
    if order_block is None or fvg is None:
        return {
            "confirmed": False,
            "low": None,
            "high": None,
            "reason": "Both zones are required.",
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
        "low": overlap_low,
        "high": overlap_high,
        "reason": "Order Block and FVG overlap.",
    }


def _build_setup_id(
    direction,
    event,
    order_block,
    fvg,
):
    raw_id = "|".join(
        [
            direction,
            str(event["event"]),
            str(event["timestamp"]),
            str(event["level"]),
            str(order_block["index"]) if order_block else "NO_OB",
            str(fvg["index"]) if fvg else "NO_FVG",
        ]
    )

    digest = hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()[:16]

    return f"BTC-USDT-4H-{direction}-{digest}"


def generate_four_hour_setup(
    candles,
    allowed_direction,
    setup_expiry_candles=24,
    ob_lookback=10,
    fvg_lookback=10,
    risk_reward=2.0,
):
    diagnostics = {
        "input_candles": 0,
        "structure_events": 0,
        "matching_events": 0,
        "order_blocks_found": 0,
        "fvg_found": 0,
        "valid_stop_swings": 0,
        "rejected_candidates": [],
    }

    if allowed_direction not in ("LONG", "SHORT"):
        return _empty_result(
            allowed_direction,
            "Allowed direction must be LONG or SHORT.",
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

    matching_events = [
        event
        for event in events
        if event.get("direction") == required_event_direction
        and event.get("event") in (
            "BOS_BULLISH",
            "CHOCH_BULLISH",
            "BOS_BEARISH",
            "CHOCH_BEARISH",
        )
        and isinstance(event.get("available_timestamp"), int)
        and event["available_timestamp"] <= current_close_timestamp
    ]

    matching_events.sort(
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
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
        event_index = event["index"]

        order_block = _find_order_block(
            validated,
            event_index,
            allowed_direction,
            ob_lookback,
        )

        fvg = _find_fvg(
            validated,
            event_index,
            allowed_direction,
            fvg_lookback,
        )

        if order_block is not None:
            diagnostics["order_blocks_found"] += 1

        if fvg is not None:
            diagnostics["fvg_found"] += 1

        if order_block is None and fvg is None:
            diagnostics["rejected_candidates"].append({
                "event": event["event"],
                "timestamp": event["timestamp"],
                "reason": "No Order Block or FVG found.",
            })
            continue

        overlap = _calculate_overlap(
            order_block,
            fvg,
        )

        if overlap["confirmed"]:
            entry_zone = {
                "source": "OB_FVG_OVERLAP",
                "low": overlap["low"],
                "high": overlap["high"],
            }

        elif order_block is not None:
            entry_zone = {
                "source": "ORDER_BLOCK",
                "low": order_block["low"],
                "high": order_block["high"],
            }

        else:
            entry_zone = {
                "source": "FVG",
                "low": fvg["low"],
                "high": fvg["high"],
            }

        entry_reference = (
            entry_zone["low"] + entry_zone["high"]
        ) / 2.0

        stop_swing = _find_stop_swing(
            swings,
            event_index,
            allowed_direction,
            entry_reference,
        )

        if stop_swing is None:
            diagnostics["rejected_candidates"].append({
                "event": event["event"],
                "timestamp": event["timestamp"],
                "reason": "No confirmed opposite swing for stop loss.",
            })
            continue

        diagnostics["valid_stop_swings"] += 1

        stop_loss = stop_swing["price"]

        if allowed_direction == "LONG":
            risk_distance = entry_reference - stop_loss
        else:
            risk_distance = stop_loss - entry_reference

        if risk_distance <= 0:
            diagnostics["rejected_candidates"].append({
                "event": event["event"],
                "timestamp": event["timestamp"],
                "reason": "Invalid risk distance.",
            })
            continue

        if allowed_direction == "LONG":
            take_profit = (
                entry_reference + risk_distance * risk_reward
            )
        else:
            take_profit = (
                entry_reference - risk_distance * risk_reward
            )

        setup_status = (
            "CONFIRMED_CONFLUENCE"
            if overlap["confirmed"]
            else "WATCH_NO_ZONE_OVERLAP"
        )

        expiry_timestamp = (
            event["available_timestamp"]
            + setup_expiry_candles * candle_duration
        )

        if current_close_timestamp > expiry_timestamp:
            setup_status = "EXPIRED"

        setup_id = _build_setup_id(
            allowed_direction,
            event,
            order_block,
            fvg,
        )

        selected_candidate = {
            "setup_id": setup_id,
            "symbol": "BTC-USDT",
            "timeframe": "4H",
            "direction": allowed_direction,
            "status": setup_status,
            "structure_event": event["event"],
            "structure_event_timestamp": event["timestamp"],
            "available_timestamp": event["available_timestamp"],
            "expiry_timestamp": expiry_timestamp,
            "entry_zone": entry_zone,
            "entry_reference": entry_reference,
            "order_block": order_block,
            "fvg": fvg,
            "zone_overlap": overlap,
            "stop_swing": {
                "type": stop_swing["type"],
                "index": stop_swing["index"],
                "timestamp": stop_swing["timestamp"],
                "confirmation_timestamp": stop_swing.get(
                    "confirmation_timestamp"
                ),
                "price": stop_swing["price"],
            },
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_distance": risk_distance,
            "risk_reward": float(risk_reward),
        }

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
        reason = "Confirmed structure event with overlapping Order Block and FVG."

    else:
        result_status = "WATCH_NO_ZONE_OVERLAP"
        reason = "A valid setup exists, but Order Block and FVG do not overlap."

    diagnostics["selected_setup_id"] = selected_candidate["setup_id"]
    diagnostics["selected_event"] = selected_candidate["structure_event"]
    diagnostics["selected_event_index"] = matching_events[
        0
    ]["index"]

    return {
        "status": result_status,
        "allowed_direction": allowed_direction,
        "reason": reason,
        "setup": selected_candidate,
        "diagnostics": diagnostics,
    }
