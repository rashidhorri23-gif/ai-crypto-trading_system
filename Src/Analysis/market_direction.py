
VALID_TRENDS = {"BULLISH", "BEARISH"}

VALID_EVENT_NAMES = {
    "BOS_BULLISH",
    "CHOCH_BULLISH",
    "BOS_BEARISH",
    "CHOCH_BEARISH",
}


def _normalize_trend(value):
    value = str(value or "").upper().strip()

    if value in ("BULLISH", "LONG"):
        return "BULLISH"

    if value in ("BEARISH", "SHORT"):
        return "BEARISH"

    return None


def _event_direction(event):
    if not isinstance(event, dict):
        return None

    event_name = str(event.get("event", "")).upper()
    direction = str(event.get("direction", "")).upper()

    if event_name not in VALID_EVENT_NAMES:
        return None

    if direction == "BULLISH" or event_name.endswith("BULLISH"):
        return "BULLISH"

    if direction == "BEARISH" or event_name.endswith("BEARISH"):
        return "BEARISH"

    return None


def extract_external_state(result):
    if not isinstance(result, dict):
        return {
            "trend": None,
            "event_bias": None,
            "reason": "Invalid structure analysis result.",
            "last_event": None,
        }

    external = result.get("external_structure") or result.get("external") or {}

    if not isinstance(external, dict):
        external = {}

    structure = external.get("structure", {})

    if not isinstance(structure, dict):
        structure = {}

    trend = _normalize_trend(
        external.get("trend")
        or structure.get("trend")
        or result.get("trend")
    )

    events = external.get("events", [])

    if not isinstance(events, list):
        events = []

    duration = result.get("candle_duration")
    last_timestamp = result.get("last_timestamp")

    last_available = None

    if (
        isinstance(duration, (int, float))
        and isinstance(last_timestamp, (int, float))
    ):
        last_available = int(last_timestamp + duration)

    valid_events = []

    for event in events:
        direction = _event_direction(event)

        if direction is None:
            continue

        try:
            available = int(event["available_timestamp"])
            event_index = int(event["index"])
        except (KeyError, TypeError, ValueError):
            continue

        if last_available is not None and available > last_available:
            continue

        valid_events.append(
            (available, event_index, direction, event)
        )

    if valid_events:
        valid_events.sort(
            key=lambda item: (item[0], item[1])
        )

        latest = valid_events[-1]
        event_bias = latest[2]
        last_event = latest[3]

    else:
        event_bias = _normalize_trend(
            external.get("event_bias")
        )
        last_event = None

    reason = (
        external.get("reason")
        or structure.get("reason")
        or "No external structure reason available."
    )

    return {
        "trend": trend,
        "event_bias": event_bias,
        "reason": reason,
        "last_event": last_event,
    }


def determine_market_direction(daily_result, four_hour_result):
    daily = extract_external_state(daily_result)
    four_hour = extract_external_state(four_hour_result)

    daily_bias = daily["trend"]
    four_hour_bias = four_hour["trend"]

    daily_event_bias = daily["event_bias"]
    four_hour_event_bias = four_hour["event_bias"]

    result = {
        "status": "WAIT",
        "allowed_direction": "WAIT",
        "trade_directions": [],
        "trade_directions_to_evaluate": [],
        "daily_bias": daily_bias or "UNKNOWN",
        "four_hour_bias": four_hour_bias or "UNKNOWN",
        "daily_event_bias": daily_event_bias or "UNKNOWN",
        "four_hour_event_bias": four_hour_event_bias or "UNKNOWN",
        "event_bias": four_hour_event_bias or "UNKNOWN",
        "daily_reason": daily["reason"],
        "four_hour_reason": four_hour["reason"],
        "daily_structure_reason": daily["reason"],
        "four_hour_structure_reason": four_hour["reason"],
        "reason": "",
    }

    if daily_bias not in VALID_TRENDS:
        result["reason"] = "Daily external trend is unclear. WAIT."
        return result

    if four_hour_bias not in VALID_TRENDS:
        result["reason"] = "4H external trend is unclear. WAIT."
        return result

    if daily_bias != four_hour_bias:
        result["reason"] = "Daily and 4H trends disagree. WAIT."
        return result

    if (
        daily_event_bias is not None
        and daily_event_bias != daily_bias
    ):
        result["reason"] = (
            "Daily event bias conflicts with daily trend. WAIT."
        )
        return result

    if (
        four_hour_event_bias is not None
        and four_hour_event_bias != four_hour_bias
    ):
        result["reason"] = (
            "4H event bias conflicts with 4H trend. WAIT."
        )
        return result

    if daily_bias == "BULLISH":
        result["status"] = "ALIGNED"
        result["allowed_direction"] = "LONG"
        result["trade_directions"] = ["LONG"]
        result["trade_directions_to_evaluate"] = ["LONG"]
        result["reason"] = (
            "Daily and 4H trends are aligned bullish."
        )

    else:
        result["status"] = "ALIGNED"
        result["allowed_direction"] = "SHORT"
        result["trade_directions"] = ["SHORT"]
        result["trade_directions_to_evaluate"] = ["SHORT"]
        result["reason"] = (
            "Daily and 4H trends are aligned bearish."
        )

    return result
