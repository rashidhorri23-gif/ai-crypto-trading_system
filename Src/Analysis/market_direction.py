
VALID_DIRECTIONS = {"BULLISH", "BEARISH"}


def normalize_direction(value):
    if value is None:
        return None

    direction = str(value).strip().upper()

    if direction in VALID_DIRECTIONS:
        return direction

    return None


def extract_external_state(result):
    if not isinstance(result, dict):
        return {
            "trend": None,
            "event_bias": None,
            "reason": "Invalid analysis result."
        }

    external = result.get("external_structure", {})

    if not isinstance(external, dict):
        external = {}

    structure = external.get("structure", {})

    if not isinstance(structure, dict):
        structure = {}

    trend_value = (
        external.get("trend")
        or structure.get("trend")
        or result.get("trend")
    )

    event_value = (
        external.get("event_bias")
        or result.get("event_bias")
    )

    reason = (
        structure.get("reason")
        or external.get("reason")
        or "No structure reason available."
    )

    return {
        "trend": normalize_direction(trend_value),
        "event_bias": normalize_direction(event_value),
        "reason": str(reason)
    }


def determine_market_direction(daily_result, four_hour_result):
    daily_state = extract_external_state(daily_result)
    four_hour_state = extract_external_state(four_hour_result)

    daily_trend = daily_state["trend"]
    daily_event_bias = daily_state["event_bias"]

    four_hour_trend = four_hour_state["trend"]
    four_hour_event_bias = four_hour_state["event_bias"]

    decision = "WAIT"
    reason = ""

    if daily_trend is None:
        reason = "Daily trend is unclear or sideways."

    elif four_hour_trend is None:
        reason = "4H external trend is unclear or sideways."

    elif (
        daily_event_bias is not None
        and daily_event_bias != daily_trend
    ):
        reason = "Daily trend and latest event bias disagree."

    elif (
        four_hour_event_bias is not None
        and four_hour_event_bias != four_hour_trend
    ):
        reason = "4H trend and latest event bias disagree."

    elif daily_trend != four_hour_trend:
        reason = "Daily and 4H directions are not aligned."

    elif daily_trend == "BULLISH":
        decision = "LONG"
        reason = "Daily and 4H structures are bullish and aligned."

    elif daily_trend == "BEARISH":
        decision = "SHORT"
        reason = "Daily and 4H structures are bearish and aligned."

    directions = []

    if decision == "LONG":
        directions = ["LONG"]
    elif decision == "SHORT":
        directions = ["SHORT"]

    return {
        "timeframe": "1D+4H",
        "daily_bias": daily_trend or "UNKNOWN",
        "daily_event_bias": daily_event_bias or "UNKNOWN",
        "four_hour_bias": four_hour_trend or "UNKNOWN",
        "four_hour_event_bias": four_hour_event_bias or "UNKNOWN",
        "allowed_direction": decision,
        "trade_directions_to_evaluate": directions,
        "status": "ALIGNED" if decision != "WAIT" else "WAIT",
        "reason": reason,
        "daily_structure_reason": daily_state["reason"],
        "four_hour_structure_reason": four_hour_state["reason"]
    }
