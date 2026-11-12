
import hashlib
import math

from Src.Analysis.daily_structure import (
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)


DEFAULT_SETUP_EXPIRY_CANDLES = 24
DEFAULT_OB_LOOKBACK = 10
DEFAULT_FVG_LOOKBACK = 10
DEFAULT_RISK_REWARD = 2.0

SYMBOL = "BTC-USDT"
TIMEFRAME = "4H"


def _empty_result(reason, direction="WAIT", diagnostics=None):
    result = {
        "timeframe": TIMEFRAME,
        "status": "WAIT",
        "allowed_direction": direction,
        "setup": None,
        "reason": reason,
    }

    if diagnostics is not None:
        result["diagnostics"] = diagnostics

    return result


def _is_valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_valid_index(value, candles_length):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < candles_length
    )


def _get_fvg_candidates(candles, event_index, direction, lookback):
    start_index = max(2, event_index - lookback + 1)
    candidates = []

    for index in range(event_index, start_index - 1, -1):
        first = candles[index - 2]
        third = candles[index]

        if direction == "LONG":
            if third["low"] > first["high"]:
                candidates.append({
                    "type": "BULLISH_FVG",
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": first["high"],
                    "high": third["low"],
                })

        elif direction == "SHORT":
            if third["high"] < first["low"]:
                candidates.append({
                    "type": "BEARISH_FVG",
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": third["high"],
                    "high": first["low"],
                })

    return candidates


def _is_fvg_fully_mitigated(
    candles,
    fvg,
    direction,
    event_index,
):
    for index in range(fvg["index"] + 1, event_index + 1):
        candle = candles[index]

        if direction == "LONG":
            if candle["low"] <= fvg["low"]:
                return True

        else:
            if candle["high"] >= fvg["high"]:
                return True

    return False


def _is_order_block_invalidated(
    candles,
    order_block,
    direction,
    event_index,
):
    for index in range(order_block["index"] + 1, event_index + 1):
        candle = candles[index]

        if direction == "LONG":
            if candle["close"] < order_block["low"]:
                return True

        else:
            if candle["close"] > order_block["high"]:
                return True

    return False


def _get_order_block_before_fvg(
    candles,
    fvg_index,
    direction,
    lookback,
    event_index,
):
    start_index = max(0, fvg_index - lookback)

    for index in range(fvg_index - 1, start_index - 1, -1):
        candle = candles[index]

        if direction == "LONG":
            if candle["close"] >= candle["open"]:
                continue

            order_block = {
                "type": "BULLISH_ORDER_BLOCK",
                "index": index,
                "timestamp": candle["timestamp"],
                "low": candle["low"],
                "high": candle["high"],
            }

        else:
            if candle["close"] <= candle["open"]:
                continue

            order_block = {
                "type": "BEARISH_ORDER_BLOCK",
                "index": index,
                "timestamp": candle["timestamp"],
                "low": candle["low"],
                "high": candle["high"],
            }

        if _is_order_block_invalidated(
            candles,
            order_block,
            direction,
            event_index,
        ):
            continue

        return order_block

    return None


def _find_valid_ob_fvg_pair(
    candles,
    event_index,
    direction,
    ob_lookback,
    fvg_lookback,
):
    fvg_candidates = _get_fvg_candidates(
        candles,
        event_index,
        direction,
        fvg_lookback,
    )

    for fvg in fvg_candidates:
        if _is_fvg_fully_mitigated(
            candles,
            fvg,
            direction,
            event_index,
        ):
            continue

        order_block = _get_order_block_before_fvg(
            candles,
            fvg["index"],
            direction,
            ob_lookback,
            event_index,
        )

        if order_block is None:
            continue

        return order_block, fvg

    return None, None


def _get_zone_overlap(order_block, fvg):
    if order_block is None or fvg is None:
        return None

    overlap_low = max(
        order_block["low"],
        fvg["low"],
    )

    overlap_high = min(
        order_block["high"],
        fvg["high"],
    )

    if overlap_low >= overlap_high:
        return None

    return {
        "low": overlap_low,
        "high": overlap_high,
        "midpoint": (overlap_low + overlap_high) / 2,
    }


def _create_setup_id(direction, event):
    raw_value = (
        f"{SYMBOL}|{TIMEFRAME}|{direction}|"
        f"{event['event']}|{event['timestamp']}|"
        f"{event['level']}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16].upper()


def _validate_event(event, candles_length):
    if not isinstance(event, dict):
        return False

    if event.get("event") not in (
        "BOS_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BULLISH",
        "CHOCH_BEARISH",
    ):
        return False

    if event.get("direction") not in ("BULLISH", "BEARISH"):
        return False

    if not _is_valid_index(event.get("index"), candles_length):
        return False

    if not _is_valid_number(event.get("timestamp")):
        return False

    if not _is_valid_number(event.get("available_timestamp")):
        return False

    if not _is_valid_number(event.get("level")):
        return False

    return True


def _get_confirmed_stop_swing(
    candles,
    swings,
    event_index,
    direction,
    entry_zone,
):
    """
    Select the latest confirmed, unbroken opposing swing
    that is also outside the complete entry zone.
    """

    required_type = "LOW" if direction == "LONG" else "HIGH"

    eligible = []
    rejected = []

    for swing in swings:
        if swing.get("type") != required_type:
            continue

        confirmation_index = swing.get("confirmation_index")
        swing_index = swing.get("index")
        swing_price = swing.get("price")
        confirmation_timestamp = swing.get("confirmation_timestamp")
        swing_timestamp = swing.get("timestamp")

        if (
            not isinstance(confirmation_index, int)
            or isinstance(confirmation_index, bool)
            or confirmation_index >= event_index
        ):
            continue

        if not _is_valid_index(swing_index, len(candles)):
            rejected.append({
                "price": swing_price,
                "reason": "INVALID_SWING_INDEX",
            })
            continue

        if not _is_valid_number(swing_price):
            rejected.append({
                "price": swing_price,
                "reason": "INVALID_SWING_PRICE",
            })
            continue

        if not _is_valid_number(confirmation_timestamp):
            rejected.append({
                "price": swing_price,
                "reason": "INVALID_CONFIRMATION_TIMESTAMP",
            })
            continue

        if not _is_valid_number(swing_timestamp):
            rejected.append({
                "price": swing_price,
                "reason": "INVALID_SWING_TIMESTAMP",
            })
            continue

        invalidated = False
        invalidation_index = None

        for index in range(confirmation_index + 1, event_index + 1):
            candle = candles[index]

            if direction == "LONG":
                if candle["low"] < swing_price:
                    invalidated = True
                    invalidation_index = index
                    break
            else:
                if candle["high"] > swing_price:
                    invalidated = True
                    invalidation_index = index
                    break

        if invalidated:
            rejected.append({
                "price": swing_price,
                "reason": "SWING_BROKEN_AFTER_CONFIRMATION",
                "invalidation_index": invalidation_index,
            })
            continue

        if direction == "SHORT":
            if swing_price <= entry_zone["high"]:
                rejected.append({
                    "price": swing_price,
                    "reason": "STOP_NOT_ABOVE_COMPLETE_ENTRY_ZONE",
                    "entry_zone_high": entry_zone["high"],
                })
                continue
        else:
            if swing_price >= entry_zone["low"]:
                rejected.append({
                    "price": swing_price,
                    "reason": "STOP_NOT_BELOW_COMPLETE_ENTRY_ZONE",
                    "entry_zone_low": entry_zone["low"],
                })
                continue

        eligible.append(swing)

    eligible.sort(
        key=lambda item: (
            item["confirmation_index"],
            item["index"],
        ),
        reverse=True,
    )

    diagnostics = {
        "required_swing_type": required_type,
        "entry_zone": {
            "low": entry_zone["low"],
            "high": entry_zone["high"],
        },
        "eligible_swing_count": len(eligible),
        "rejected_swing_count": len(rejected),
        "rejected_swings": rejected,
    }

    if not eligible:
        return None, diagnostics

    selected = eligible[0]

    diagnostics["selected_swing"] = {
        "type": selected["type"],
        "index": selected["index"],
        "price": selected["price"],
        "timestamp": selected["timestamp"],
        "confirmation_index": selected["confirmation_index"],
        "confirmation_timestamp": selected["confirmation_timestamp"],
    }

    return selected, diagnostics


def generate_four_hour_setup(
    candles,
    allowed_direction,
    setup_expiry_candles=DEFAULT_SETUP_EXPIRY_CANDLES,
    ob_lookback=DEFAULT_OB_LOOKBACK,
    fvg_lookback=DEFAULT_FVG_LOOKBACK,
    risk_reward=DEFAULT_RISK_REWARD,
):
    if allowed_direction not in ("LONG", "SHORT", "BOTH", "WAIT"):
        raise ValueError("Invalid allowed_direction")

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
            raise ValueError(f"{name} must be a positive integer")

    if not _is_valid_number(risk_reward) or risk_reward <= 0:
        raise ValueError("risk_reward must be a finite positive number")

    if allowed_direction == "WAIT":
        return _empty_result(
            "Higher-timeframe direction does not authorize a setup."
        )

    candles = validate_candles(candles)

    if not isinstance(candles, list):
        raise ValueError("Validated candles must be a list")

    candle_duration = infer_candle_duration(candles)

    if not _is_valid_number(candle_duration) or candle_duration <= 0:
        raise ValueError("Invalid candle duration")

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

    if not events:
        return _empty_result(
            "No confirmed 4H BOS or CHOCH event was found.",
            allowed_direction,
        )

    valid_events = [
        event for event in events
        if _validate_event(event, len(candles))
    ]

    if not valid_events:
        return _empty_result(
            "No valid confirmed structure event was found.",
            allowed_direction,
        )

    valid_events.sort(
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
        )
    )

    latest_event = valid_events[-1]

    event_direction = (
        "LONG"
        if latest_event["direction"] == "BULLISH"
        else "SHORT"
    )

    if (
        allowed_direction != "BOTH"
        and event_direction != allowed_direction
    ):
        return _empty_result(
            "Latest 4H structure event conflicts with the authorized direction.",
            allowed_direction,
        )

    event_index = latest_event["index"]

    latest_candle_close = (
        candles[-1]["timestamp"] + candle_duration
    )

    event_available_timestamp = latest_event["available_timestamp"]

    if event_available_timestamp > latest_candle_close:
        return _empty_result(
            "Structure event is not yet confirmed by a completed candle.",
            allowed_direction,
        )

    expiry_timestamp = (
        event_available_timestamp
        + setup_expiry_candles * candle_duration
    )

    if latest_candle_close >= expiry_timestamp:
        return _empty_result(
            "The latest structure event has expired.",
            allowed_direction,
        )

    order_block, fvg = _find_valid_ob_fvg_pair(
        candles=candles,
        event_index=event_index,
        direction=event_direction,
        ob_lookback=ob_lookback,
        fvg_lookback=fvg_lookback,
    )

    if order_block is None or fvg is None:
        return _empty_result(
            "No valid Order Block linked to an unmitigated FVG was found before the structure break.",
            allowed_direction,
            {
                "event": latest_event,
                "order_block": order_block,
                "fvg": fvg,
            },
        )

    overlap_zone = _get_zone_overlap(order_block, fvg)

    if overlap_zone is not None:
        entry_zone = {
            "low": overlap_zone["low"],
            "high": overlap_zone["high"],
            "source": "ORDER_BLOCK_FVG_OVERLAP",
        }
        setup_status = "CONFIRMED_CONFLUENCE"
    else:
        entry_zone = {
            "low": order_block["low"],
            "high": order_block["high"],
            "source": "ORDER_BLOCK",
        }
        setup_status = "WATCH_NO_ZONE_OVERLAP"

    entry_price = (
        entry_zone["low"] + entry_zone["high"]
    ) / 2

    stop_swing, stop_diagnostics = _get_confirmed_stop_swing(
        candles=candles,
        swings=swings,
        event_index=event_index,
        direction=event_direction,
        entry_zone=entry_zone,
    )

    if stop_swing is None:
        return _empty_result(
            "No confirmed and unbroken opposing swing exists outside the complete entry zone.",
            allowed_direction,
            {
                "event": latest_event,
                "order_block": order_block,
                "fvg": fvg,
                "zone_overlap": overlap_zone,
                "stop_diagnostics": stop_diagnostics,
            },
        )

    stop_loss = stop_swing["price"]

    if event_direction == "LONG":
        risk_distance = entry_price - stop_loss
        take_profit = entry_price + risk_distance * risk_reward
    else:
        risk_distance = stop_loss - entry_price
        take_profit = entry_price - risk_distance * risk_reward

    if not _is_valid_number(risk_distance) or risk_distance <= 0:
        return _empty_result(
            "Invalid risk distance.",
            allowed_direction,
            {
                "entry_zone": entry_zone,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "stop_diagnostics": stop_diagnostics,
            },
        )

    if not _is_valid_number(take_profit) or take_profit <= 0:
        return _empty_result(
            "Invalid take profit price.",
            allowed_direction,
            {
                "entry_zone": entry_zone,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "risk_distance": risk_distance,
            },
        )

    setup = {
        "setup_id": _create_setup_id(
            event_direction,
            latest_event,
        ),
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "direction": event_direction,
        "status": setup_status,
        "structure_event": latest_event["event"],
        "structure_event_timestamp": latest_event["timestamp"],
        "available_timestamp": event_available_timestamp,
        "expiry_timestamp": expiry_timestamp,
        "entry_zone": entry_zone,
        "entry_reference": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_distance": risk_distance,
        "risk_reward": risk_reward,
        "order_block": order_block,
        "fvg": fvg,
        "zone_overlap": overlap_zone,
        "stop_swing": {
            "type": stop_swing["type"],
            "price": stop_swing["price"],
            "timestamp": stop_swing["timestamp"],
            "confirmation_timestamp": stop_swing["confirmation_timestamp"],
        },
        "setup_expiry_candles": setup_expiry_candles,
        "candle_duration_seconds": candle_duration,
        "entry_confirmation_required": True,
        "live_trading": False,
    }

    if setup_status == "WATCH_NO_ZONE_OVERLAP":
        reason = (
            "A valid Order Block and FVG were found, "
            "but their price zones do not overlap. "
            "The setup is watch-only."
        )
    else:
        reason = (
            "The 4H structure event, Order Block, FVG, "
            "zone overlap and opposing swing are valid. "
            "The setup is ready for lower-timeframe confirmation."
        )

    return {
        "timeframe": TIMEFRAME,
        "status": setup_status,
        "allowed_direction": allowed_direction,
        "setup": setup,
        "reason": reason,
        "diagnostics": {
            "stop_diagnostics": stop_diagnostics,
        },
    }
