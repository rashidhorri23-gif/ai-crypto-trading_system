
from typing import Any, Dict, List, Optional, Tuple

from Src.Analysis.daily_structure import analyze_daily_structure


TIMEFRAME_PRIORITY = {
    "1H": 0,
    "15m": 1,
    "5m": 2,
}

ALLOWED_EVENT_TYPES = {"BOS", "CHOCH"}


def _validate_candles(candles: List[Dict[str, Any]]) -> None:
    if not isinstance(candles, list) or len(candles) < 20:
        raise ValueError("At least 20 candles are required.")

    required_fields = ("timestamp", "open", "high", "low", "close")
    previous_timestamp = None

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError("Invalid candle at index {}.".format(index))

        for field in required_fields:
            if field not in candle:
                raise ValueError(
                    "Missing candle field '{}' at index {}.".format(
                        field, index
                    )
                )

        timestamp = int(candle["timestamp"])
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])

        if timestamp <= 0:
            raise ValueError("Invalid candle timestamp.")

        if min(open_price, high_price, low_price, close_price) <= 0:
            raise ValueError("Candle prices must be positive.")

        if high_price < max(open_price, close_price, low_price):
            raise ValueError("Invalid candle high at index {}.".format(index))

        if low_price > min(open_price, close_price, high_price):
            raise ValueError("Invalid candle low at index {}.".format(index))

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Candle timestamps are not strictly increasing.")

        previous_timestamp = timestamp


def _collect_events(
    value: Any,
    collected: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    if collected is None:
        collected = []

    if isinstance(value, dict):
        if (
            "event" in value
            and "index" in value
            and "available_timestamp" in value
        ):
            collected.append(value)

        for nested_value in value.values():
            if isinstance(nested_value, (dict, list)):
                _collect_events(nested_value, collected)

    elif isinstance(value, list):
        for item in value:
            _collect_events(item, collected)

    return collected


def _get_event_direction(event: Dict[str, Any]) -> Optional[str]:
    direction = str(event.get("direction", "")).upper()

    if direction in ("BULLISH", "LONG"):
        return "LONG"

    if direction in ("BEARISH", "SHORT"):
        return "SHORT"

    event_name = str(event.get("event", "")).upper()

    if event_name.endswith("BULLISH"):
        return "LONG"

    if event_name.endswith("BEARISH"):
        return "SHORT"

    return None


def _is_allowed_structure_event(event: Dict[str, Any]) -> bool:
    event_name = str(event.get("event", "")).upper()

    return any(
        event_name.startswith(event_type + "_")
        for event_type in ALLOWED_EVENT_TYPES
    )


def _find_next_candle(
    candles: List[Dict[str, Any]],
    event_index: int
) -> Optional[Dict[str, Any]]:
    next_index = event_index + 1

    if next_index >= len(candles):
        return None

    return candles[next_index]


def _empty_result(
    status: str,
    reason: str
) -> Dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "entry_signal": False,
        "setup_id": None,
        "direction": "WAIT",
        "confirmation_timeframe": None,
        "confirmation_event": None,
        "confirmation_timestamp": None,
        "entry_timestamp": None,
        "entry_price": None,
        "live_trading": False,
    }


def confirm_lower_timeframe_entry(
    setup: Dict[str, Any],
    candles_by_timeframe: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    if not isinstance(setup, dict):
        return _empty_result("WAIT", "Invalid 4H setup input.")

    setup_status = str(setup.get("status", "")).upper()

    if setup_status != "CONFIRMED_CONFLUENCE":
        return _empty_result(
            "WAIT",
            "The 4H setup is not confirmed. Lower-timeframe entry checks are blocked."
        )

    setup_id = setup.get("setup_id")

    if not setup_id:
        return _empty_result("WAIT", "The 4H setup has no setup ID.")

    direction = str(setup.get("direction", "")).upper()

    if direction not in ("LONG", "SHORT"):
        return _empty_result("WAIT", "The 4H setup direction is invalid.")

    try:
        setup_available_timestamp = int(setup["available_timestamp"])
        expiry_timestamp = int(setup["expiry_timestamp"])
    except (KeyError, TypeError, ValueError):
        return _empty_result(
            "WAIT",
            "The 4H setup timestamps are missing or invalid."
        )

    candidates: List[Tuple[int, int, str, Dict[str, Any], Dict[str, Any]]] = []

    for timeframe in ("1H", "15m", "5m"):
        candles = candles_by_timeframe.get(timeframe, [])

        if not isinstance(candles, list) or len(candles) < 20:
            continue

        try:
            _validate_candles(candles)

            analysis = analyze_daily_structure(
                candles,
                timeframe=timeframe
            )

            events = _collect_events(analysis)

        except Exception:
            continue

        for event in events:
            if not _is_allowed_structure_event(event):
                continue

            event_direction = _get_event_direction(event)

            if event_direction != direction:
                continue

            try:
                event_index = int(event["index"])
                available_timestamp = int(event["available_timestamp"])
            except (TypeError, ValueError, KeyError):
                continue

            if available_timestamp < setup_available_timestamp:
                continue

            if available_timestamp >= expiry_timestamp:
                continue

            if event_index < 0 or event_index >= len(candles):
                continue

            entry_candle = _find_next_candle(candles, event_index)

            if entry_candle is None:
                continue

            entry_timestamp = int(entry_candle["timestamp"])

            if entry_timestamp < available_timestamp:
                continue

            if entry_timestamp >= expiry_timestamp:
                continue

            entry_price = float(entry_candle["open"])

            if entry_price <= 0:
                continue

            candidates.append(
                (
                    available_timestamp,
                    TIMEFRAME_PRIORITY[timeframe],
                    timeframe,
                    event,
                    entry_candle,
                )
            )

    if not candidates:
        return {
            "status": "WAITING_FOR_CONFIRMATION",
            "reason": (
                "No aligned BOS or CHOCH confirmation was found "
                "after the 4H setup became available."
            ),
            "entry_signal": False,
            "setup_id": setup_id,
            "direction": direction,
            "confirmation_timeframe": None,
            "confirmation_event": None,
            "confirmation_timestamp": None,
            "entry_timestamp": None,
            "entry_price": None,
            "live_trading": False,
        }

    candidates.sort(key=lambda item: (item[0], item[1]))

    (
        confirmation_timestamp,
        _,
        confirmation_timeframe,
        confirmation_event,
        entry_candle,
    ) = candidates[0]

    entry_timestamp = int(entry_candle["timestamp"])
    entry_price = float(entry_candle["open"])

    return {
        "status": "ENTRY_CONFIRMED",
        "reason": (
            "The first aligned lower-timeframe BOS or CHOCH "
            "was confirmed after the 4H setup became available."
        ),
        "entry_signal": True,
        "setup_id": setup_id,
        "direction": direction,
        "confirmation_timeframe": confirmation_timeframe,
        "confirmation_event": confirmation_event.get("event"),
        "confirmation_timestamp": confirmation_timestamp,
        "entry_timestamp": entry_timestamp,
        "entry_price": entry_price,
        "live_trading": False,
    }
