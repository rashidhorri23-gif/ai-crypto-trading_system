
from typing import Any, Dict, List, Optional

from Src.Analysis.daily_structure import (
    validate_candles,
    infer_candle_duration,
    detect_swings,
    analyze_structure_events,
)


TIMEFRAME_PRIORITY = {
    "1H": 0,
    "15m": 1,
    "5m": 2,
}

VALID_EVENTS = {
    "BOS_BULLISH",
    "CHOCH_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BEARISH",
}


def _empty_result(status, reason, setup_id=None, direction="WAIT"):
    return {
        "status": status,
        "reason": reason,
        "entry_signal": False,
        "setup_id": setup_id,
        "direction": direction,
        "confirmation_timeframe": None,
        "confirmation_event": None,
        "confirmation_timestamp": None,
        "retest_timestamp": None,
        "entry_timestamp": None,
        "entry_price": None,
        "live_trading": False,
    }


def _normalize_setup(setup):
    if not isinstance(setup, dict):
        return None

    nested = setup.get("setup")

    if isinstance(nested, dict):
        return nested

    return setup


def _validate_setup(setup):
    if not isinstance(setup, dict):
        raise ValueError("Invalid 4H setup.")

    if setup.get("status") != "CONFIRMED_CONFLUENCE":
        raise ValueError("4H setup is not confirmed.")

    if not setup.get("setup_id"):
        raise ValueError("Missing setup ID.")

    if setup.get("direction") not in ("LONG", "SHORT"):
        raise ValueError("Invalid setup direction.")

    zone = setup.get("entry_zone")

    if not isinstance(zone, dict):
        raise ValueError("Missing 4H entry zone.")

    low = zone.get("low")
    high = zone.get("high")

    if (
        not isinstance(low, (int, float))
        or isinstance(low, bool)
        or not isinstance(high, (int, float))
        or isinstance(high, bool)
        or low <= 0
        or high <= low
    ):
        raise ValueError("Invalid 4H entry zone.")

    available = setup.get("available_timestamp")
    expiry = setup.get("expiry_timestamp")

    if (
        not isinstance(available, (int, float))
        or isinstance(available, bool)
        or not isinstance(expiry, (int, float))
        or isinstance(expiry, bool)
        or available <= 0
        or expiry <= available
    ):
        raise ValueError("Invalid setup timestamps.")

    return {
        "setup_id": str(setup["setup_id"]),
        "direction": setup["direction"],
        "zone_low": float(low),
        "zone_high": float(high),
        "available_timestamp": int(available),
        "expiry_timestamp": int(expiry),
    }


def _direction_for_event(event):
    name = str(event.get("event", "")).upper()
    direction = str(event.get("direction", "")).upper()

    if name not in VALID_EVENTS:
        return None

    if direction == "BULLISH" and name.endswith("BULLISH"):
        return "LONG"

    if direction == "BEARISH" and name.endswith("BEARISH"):
        return "SHORT"

    return None


def _find_retest(candles, duration, setup_data):
    zone_low = setup_data["zone_low"]
    zone_high = setup_data["zone_high"]
    available = setup_data["available_timestamp"]
    expiry = setup_data["expiry_timestamp"]

    for index, candle in enumerate(candles):
        candle_open = candle["timestamp"]
        candle_close = candle_open + duration

        if candle_close <= available:
            continue

        if candle_open >= expiry:
            break

        intersects_zone = (
            candle["high"] >= zone_low
            and candle["low"] <= zone_high
        )

        if not intersects_zone:
            continue

        return {
            "index": index,
            "timestamp": candle_open,
            "close_timestamp": candle_close,
        }

    return None


def _find_confirmation(
    candles,
    events,
    duration,
    setup_data,
    retest,
):
    direction = setup_data["direction"]
    expiry = setup_data["expiry_timestamp"]

    candidates = []

    for event in events:
        if _direction_for_event(event) != direction:
            continue

        try:
            event_index = int(event["index"])
            available_timestamp = int(event["available_timestamp"])
        except (KeyError, TypeError, ValueError):
            continue

        if event_index <= retest["index"]:
            continue

        if event_index < 0 or event_index >= len(candles):
            continue

        if available_timestamp < setup_data["available_timestamp"]:
            continue

        if available_timestamp >= expiry:
            continue

        if available_timestamp != (
            candles[event_index]["timestamp"] + duration
        ):
            continue

        if available_timestamp <= retest["close_timestamp"]:
            continue

        next_index = event_index + 1

        if next_index >= len(candles):
            candidates.append({
                "event": event,
                "event_index": event_index,
                "available_timestamp": available_timestamp,
                "entry_candle": None,
            })
            continue

        entry_candle = candles[next_index]

        if entry_candle["timestamp"] < available_timestamp:
            continue

        if entry_candle["timestamp"] >= expiry:
            continue

        candidates.append({
            "event": event,
            "event_index": event_index,
            "available_timestamp": available_timestamp,
            "entry_candle": entry_candle,
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["available_timestamp"],
            item["event_index"],
        )
    )

    return candidates[0]


def confirm_lower_timeframe_entry(
    setup: Dict[str, Any],
    candles_by_timeframe: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    normalized = _normalize_setup(setup)

    try:
        setup_data = _validate_setup(normalized)
    except Exception as error:
        return _empty_result(
            "WAIT",
            "Invalid 4H setup: {}".format(error),
        )

    setup_id = setup_data["setup_id"]
    direction = setup_data["direction"]

    if not isinstance(candles_by_timeframe, dict):
        return _empty_result(
            "WAIT",
            "Invalid lower-timeframe candle input.",
            setup_id,
            direction,
        )

    candidates = []
    pending_entry = []

    for timeframe in ("1H", "15m", "5m"):
        candles = candles_by_timeframe.get(timeframe, [])

        if not isinstance(candles, list) or len(candles) < 20:
            continue

        try:
            candles = validate_candles(candles)
            duration = infer_candle_duration(candles)

            swings = detect_swings(
                candles,
                left_bars=2,
                right_bars=2,
                candle_duration=duration,
            )

            events = analyze_structure_events(
                candles,
                swings,
                duration,
            )

        except Exception:
            continue

        retest = _find_retest(
            candles,
            duration,
            setup_data,
        )

        if retest is None:
            continue

        confirmation = _find_confirmation(
            candles,
            events,
            duration,
            setup_data,
            retest,
        )

        if confirmation is None:
            continue

        candidate = {
            "timeframe": timeframe,
            "priority": TIMEFRAME_PRIORITY[timeframe],
            "retest": retest,
            "confirmation": confirmation,
        }

        if confirmation["entry_candle"] is None:
            pending_entry.append(candidate)
        else:
            candidates.append(candidate)

    if not candidates:
        if pending_entry:
            pending_entry.sort(
                key=lambda item: (
                    item["confirmation"]["available_timestamp"],
                    item["priority"],
                )
            )

            selected = pending_entry[0]
            confirmation = selected["confirmation"]

            return {
                "status": "WAITING_FOR_ENTRY_CANDLE",
                "reason": (
                    "The aligned structure event is confirmed. "
                    "Waiting for the next lower-timeframe candle open."
                ),
                "entry_signal": False,
                "setup_id": setup_id,
                "direction": direction,
                "confirmation_timeframe": selected["timeframe"],
                "confirmation_event": confirmation["event"]["event"],
                "confirmation_timestamp": confirmation[
                    "available_timestamp"
                ],
                "retest_timestamp": selected["retest"]["timestamp"],
                "entry_timestamp": None,
                "entry_price": None,
                "live_trading": False,
            }

        return _empty_result(
            "WAITING_FOR_CONFIRMATION",
            (
                "Waiting for price to return to the 4H entry zone "
                "and then form an aligned BOS or CHOCH."
            ),
            setup_id,
            direction,
        )

    candidates.sort(
        key=lambda item: (
            item["confirmation"]["available_timestamp"],
            item["priority"],
        )
    )

    selected = candidates[0]
    confirmation = selected["confirmation"]
    entry_candle = confirmation["entry_candle"]

    return {
        "status": "ENTRY_CONFIRMED",
        "reason": (
            "Price retested the 4H entry zone and the first aligned "
            "lower-timeframe BOS or CHOCH was confirmed."
        ),
        "entry_signal": True,
        "setup_id": setup_id,
        "direction": direction,
        "confirmation_timeframe": selected["timeframe"],
        "confirmation_event": confirmation["event"]["event"],
        "confirmation_timestamp": confirmation[
            "available_timestamp"
        ],
        "retest_timestamp": selected["retest"]["timestamp"],
        "entry_timestamp": int(entry_candle["timestamp"]),
        "entry_price": float(entry_candle["open"]),
        "live_trading": False,
    }
