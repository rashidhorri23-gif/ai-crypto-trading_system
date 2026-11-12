
import hashlib

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


def _empty_result(reason, direction="WAIT"):
    return {
        "timeframe": "4H",
        "status": "WAIT",
        "allowed_direction": direction,
        "setup": None,
        "reason": reason,
    }


def _get_order_block(candles, event_index, direction, lookback):
    start_index = max(0, event_index - lookback)

    for index in range(event_index - 1, start_index - 1, -1):
        candle = candles[index]

        if direction == "LONG" and candle["close"] < candle["open"]:
            return {
                "type": "BULLISH_ORDER_BLOCK",
                "index": index,
                "timestamp": candle["timestamp"],
                "low": candle["low"],
                "high": candle["high"],
            }

        if direction == "SHORT" and candle["close"] > candle["open"]:
            return {
                "type": "BEARISH_ORDER_BLOCK",
                "index": index,
                "timestamp": candle["timestamp"],
                "low": candle["low"],
                "high": candle["high"],
            }

    return None


def _get_fvg(candles, event_index, direction, lookback):
    start_index = max(2, event_index - lookback)

    for index in range(event_index, start_index - 1, -1):
        first = candles[index - 2]
        third = candles[index]

        if direction == "LONG":
            if third["low"] > first["high"]:
                return {
                    "type": "BULLISH_FVG",
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": first["high"],
                    "high": third["low"],
                }

        elif direction == "SHORT":
            if third["high"] < first["low"]:
                return {
                    "type": "BEARISH_FVG",
                    "index": index,
                    "timestamp": third["timestamp"],
                    "low": third["high"],
                    "high": first["low"],
                }

    return None


def _get_confirmed_stop_swing(
    swings,
    event_index,
    direction,
):
    required_type = "LOW" if direction == "LONG" else "HIGH"

    eligible = [
        swing
        for swing in swings
        if (
            swing["type"] == required_type
            and swing["confirmation_index"] < event_index
        )
    ]

    if not eligible:
        return None

    eligible.sort(
        key=lambda swing: (
            swing["confirmation_index"],
            swing["index"],
        )
    )

    return eligible[-1]


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
        f"BTC-USDT|4H|{direction}|"
        f"{event['event']}|{event['timestamp']}|"
        f"{event['level']}"
    )

    return hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16].upper()


def generate_four_hour_setup(
    candles,
    allowed_direction,
    setup_expiry_candles=DEFAULT_SETUP_EXPIRY_CANDLES,
    ob_lookback=DEFAULT_OB_LOOKBACK,
    fvg_lookback=DEFAULT_FVG_LOOKBACK,
    risk_reward=DEFAULT_RISK_REWARD,
):
    if allowed_direction not in (
        "LONG",
        "SHORT",
        "BOTH",
        "WAIT",
    ):
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
            raise ValueError(
                f"{name} must be a positive integer"
            )

    if risk_reward <= 0:
        raise ValueError("risk_reward must be positive")

    if allowed_direction == "WAIT":
        return _empty_result(
            "Higher-timeframe direction does not authorize a setup."
        )

    candles = validate_candles(candles)

    candle_duration = infer_candle_duration(candles)

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

    latest_event = events[-1]

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

    if event_index >= len(candles):
        return _empty_result(
            "Structure event index is outside the candle archive.",
            allowed_direction,
        )

    latest_candle_close = (
        candles[-1]["timestamp"] + candle_duration
    )

    event_available_timestamp = (
        latest_event["available_timestamp"]
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

    order_block = _get_order_block(
        candles,
        event_index,
        event_direction,
        ob_lookback,
    )

    fvg = _get_fvg(
        candles,
        event_index,
        event_direction,
        fvg_lookback,
    )

    if order_block is None:
        return _empty_result(
            "No valid opposite-candle Order Block was found before the structure break.",
            allowed_direction,
        )

    if fvg is None:
        return _empty_result(
            "No valid FVG was found before or at the structure break.",
            allowed_direction,
        )

    overlap_zone = _get_zone_overlap(
        order_block,
        fvg,
    )

    stop_swing = _get_confirmed_stop_swing(
        swings,
        event_index,
        event_direction,
    )

    if stop_swing is None:
        return _empty_result(
            "No confirmed opposing swing is available for stop placement.",
            allowed_direction,
        )

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

    if event_direction == "LONG":
        stop_loss = stop_swing["price"]

        if stop_loss >= entry_price:
            return _empty_result(
                "Invalid LONG setup: stop loss is not below the entry zone.",
                allowed_direction,
            )

        risk_distance = entry_price - stop_loss
        take_profit = (
            entry_price + risk_distance * risk_reward
        )

    else:
        stop_loss = stop_swing["price"]

        if stop_loss <= entry_price:
            return _empty_result(
                "Invalid SHORT setup: stop loss is not above the entry zone.",
                allowed_direction,
            )

        risk_distance = stop_loss - entry_price
        take_profit = (
            entry_price - risk_distance * risk_reward
        )

    if risk_distance <= 0:
        return _empty_result(
            "Invalid risk distance.",
            allowed_direction,
        )

    setup = {
        "setup_id": _create_setup_id(
            event_direction,
            latest_event,
        ),
        "symbol": "BTC-USDT",
        "timeframe": "4H",
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
            "confirmation_timestamp": (
                stop_swing["confirmation_timestamp"]
            ),
        },
        "setup_expiry_candles": setup_expiry_candles,
        "candle_duration_seconds": candle_duration,
        "entry_confirmation_required": True,
        "live_trading": False,
    }

    if setup_status == "WATCH_NO_ZONE_OVERLAP":
        reason = (
            "Order Block and FVG were found, but their price zones do not overlap. "
            "The setup is watch-only and is not ready for entry confirmation."
        )
    else:
        reason = (
            "4H structure, Order Block, FVG and opposing swing are aligned. "
            "The setup is ready for lower-timeframe entry confirmation."
        )

    return {
        "timeframe": "4H",
        "status": setup_status,
        "allowed_direction": allowed_direction,
        "setup": setup,
        "reason": reason,
    }
