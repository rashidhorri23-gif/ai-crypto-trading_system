
import math
from statistics import median


def _is_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_integer(value):
    return _is_number(value) and int(value) == value


def validate_candles(candles):
    if not isinstance(candles, list) or len(candles) < 2:
        raise ValueError("At least two candles are required.")

    result = []
    previous_timestamp = None

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError("Invalid candle at index {}.".format(index))

        for field in ("timestamp", "open", "high", "low", "close"):
            if not _is_number(candle.get(field)):
                raise ValueError(
                    "Invalid {} at candle index {}.".format(field, index)
                )

        timestamp = candle["timestamp"]
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])

        if not _is_integer(timestamp) or timestamp < 0:
            raise ValueError("Invalid candle timestamp.")

        if min(open_price, high_price, low_price, close_price) <= 0:
            raise ValueError("Candle prices must be positive.")

        if high_price < max(open_price, close_price, low_price):
            raise ValueError("Invalid candle high at index {}.".format(index))

        if low_price > min(open_price, close_price, high_price):
            raise ValueError("Invalid candle low at index {}.".format(index))

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Candle timestamps are not strictly increasing.")

        normalized = dict(candle)
        normalized.update({
            "timestamp": int(timestamp),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
        })

        if candle.get("volume") is not None:
            if not _is_number(candle["volume"]) or candle["volume"] < 0:
                raise ValueError("Invalid candle volume.")
            normalized["volume"] = float(candle["volume"])

        result.append(normalized)
        previous_timestamp = timestamp

    return result


def infer_candle_duration(candles):
    if not isinstance(candles, list) or len(candles) < 2:
        raise ValueError("At least two candles are required.")

    differences = [
        candles[index]["timestamp"] - candles[index - 1]["timestamp"]
        for index in range(1, len(candles))
    ]

    if any(value <= 0 for value in differences):
        raise ValueError("Invalid candle timestamp sequence.")

    duration = int(median(differences))

    if duration <= 0:
        raise ValueError("Invalid candle duration.")

    return duration


def was_swing_broken_before_confirmation(candles, swing):
    pivot = int(swing["index"])
    confirmation = int(swing["confirmation_index"])
    price = float(swing["price"])

    for index in range(pivot + 1, confirmation + 1):
        close_price = candles[index]["close"]

        if swing["type"] == "HIGH" and close_price > price:
            return True

        if swing["type"] == "LOW" and close_price < price:
            return True

    return False


def detect_swings(
    candles,
    left_bars=3,
    right_bars=3,
    candle_duration=None,
):
    candles = validate_candles(candles)

    if (
        isinstance(left_bars, bool)
        or not isinstance(left_bars, int)
        or left_bars < 1
        or isinstance(right_bars, bool)
        or not isinstance(right_bars, int)
        or right_bars < 1
    ):
        raise ValueError("Swing windows must be positive integers.")

    if candle_duration is None:
        candle_duration = infer_candle_duration(candles)

    if not _is_integer(candle_duration) or candle_duration <= 0:
        raise ValueError("Invalid candle duration.")

    swings = []

    for index in range(left_bars, len(candles) - right_bars):
        current = candles[index]
        previous = candles[index - left_bars:index]
        following = candles[index + 1:index + right_bars + 1]

        is_high = (
            current["high"] > max(item["high"] for item in previous)
            and current["high"] >= max(item["high"] for item in following)
        )

        is_low = (
            current["low"] < min(item["low"] for item in previous)
            and current["low"] <= min(item["low"] for item in following)
        )

        if not is_high and not is_low:
            continue

        confirmation_index = index + right_bars
        confirmation_timestamp = (
            candles[confirmation_index]["timestamp"] + candle_duration
        )

        for swing_type, condition, price in (
            ("HIGH", is_high, current["high"]),
            ("LOW", is_low, current["low"]),
        ):
            if not condition:
                continue

            swing = {
                "index": index,
                "timestamp": current["timestamp"],
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": confirmation_timestamp,
                "ambiguous_same_candle": is_high and is_low,
                "type": swing_type,
                "price": float(price),
            }

            if not was_swing_broken_before_confirmation(candles, swing):
                swings.append(swing)

    swings.sort(
        key=lambda item: (
            item["confirmation_index"],
            item["index"],
            item["type"],
        )
    )

    return swings


def _valid_swings(swings, candle_count):
    result = []

    for swing in swings:
        if not isinstance(swing, dict):
            continue

        if swing.get("ambiguous_same_candle"):
            continue

        swing_type = swing.get("type")
        index = swing.get("index")
        confirmation_index = swing.get("confirmation_index")
        price = swing.get("price")

        if swing_type not in ("HIGH", "LOW"):
            continue

        if (
            not _is_integer(index)
            or not _is_integer(confirmation_index)
            or not _is_number(price)
            or price <= 0
        ):
            continue

        index = int(index)
        confirmation_index = int(confirmation_index)

        if (
            index < 0
            or confirmation_index <= index
            or confirmation_index >= candle_count
        ):
            continue

        result.append(swing)

    return result


def _compare_swings(swings, swing_type):
    selected = sorted(
        [item for item in swings if item["type"] == swing_type],
        key=lambda item: (
            item["confirmation_index"],
            item["index"],
        ),
    )

    if len(selected) < 2:
        return None, selected[-1] if selected else None

    previous_price = selected[-2]["price"]
    latest_price = selected[-1]["price"]

    if latest_price > previous_price:
        comparison = "HH" if swing_type == "HIGH" else "HL"
    elif latest_price < previous_price:
        comparison = "LH" if swing_type == "HIGH" else "LL"
    else:
        comparison = "EQH" if swing_type == "HIGH" else "EQL"

    return comparison, selected[-1]


def classify_market_structure(candles, swings):
    candles = validate_candles(candles)
    swings = _valid_swings(swings, len(candles))

    high_comparison, latest_high = _compare_swings(swings, "HIGH")
    low_comparison, latest_low = _compare_swings(swings, "LOW")

    if high_comparison == "HH" and low_comparison == "HL":
        trend = "BULLISH"
        reason = "Confirmed higher highs and higher lows."

    elif high_comparison == "LH" and low_comparison == "LL":
        trend = "BEARISH"
        reason = "Confirmed lower highs and lower lows."

    else:
        trend = "SIDEWAYS"

        if high_comparison is None or low_comparison is None:
            reason = "Insufficient confirmed swing structure."
        else:
            reason = "High and low swing structures are not aligned."

    return {
        "trend": trend,
        "reason": reason,
        "high_structure": high_comparison,
        "low_structure": low_comparison,
        "latest_swing_high": latest_high,
        "latest_swing_low": latest_low,
        "swing_high_count": sum(
            1 for item in swings if item["type"] == "HIGH"
        ),
        "swing_low_count": sum(
            1 for item in swings if item["type"] == "LOW"
        ),
        "last_close": candles[-1]["close"],
        "last_timestamp": candles[-1]["timestamp"],
    }


def _swing_identity(swing):
    return (
        swing["type"],
        swing["index"],
        swing["confirmation_index"],
        swing["price"],
    )


def analyze_structure_events(candles, swings, candle_duration):
    candles = validate_candles(candles)

    if not _is_integer(candle_duration) or candle_duration <= 0:
        raise ValueError("Invalid candle duration.")

    swings = _valid_swings(swings, len(candles))
    events = []
    consumed = set()
    market_bias = None

    for candle_index, candle in enumerate(candles):
        available = [
            swing for swing in swings
            if (
                swing["confirmation_index"] < candle_index
                and _swing_identity(swing) not in consumed
            )
        ]

        crossed = []

        for swing in available:
            if swing["type"] == "HIGH" and candle["close"] > swing["price"]:
                crossed.append((swing, "BULLISH"))

            elif swing["type"] == "LOW" and candle["close"] < swing["price"]:
                crossed.append((swing, "BEARISH"))

        if not crossed:
            continue

        for swing, _ in crossed:
            consumed.add(_swing_identity(swing))

        directions = {direction for _, direction in crossed}

        crossed_levels = [
            {
                "swing_type": swing["type"],
                "direction": direction,
                "level": swing["price"],
                "swing_index": swing["index"],
                "swing_timestamp": swing["timestamp"],
                "swing_confirmation_index": swing["confirmation_index"],
            }
            for swing, direction in crossed
        ]

        crossed.sort(
            key=lambda item: (
                item[0]["confirmation_index"],
                item[0]["index"],
            ),
            reverse=True,
        )

        available_timestamp = candle["timestamp"] + candle_duration

        if len(directions) > 1:
            events.append({
                "event": "AMBIGUOUS_BREAK",
                "event_type": "AMBIGUOUS",
                "direction": "UNKNOWN",
                "index": candle_index,
                "timestamp": candle["timestamp"],
                "available_timestamp": available_timestamp,
                "crossed_levels": crossed_levels,
                "ambiguous_break": True,
            })
            continue

        selected, direction = crossed[0]

        event_type = (
            "BOS"
            if market_bias is None or direction == market_bias
            else "CHOCH"
        )

        events.append({
            "event": "{}_{}".format(event_type, direction),
            "event_type": event_type,
            "direction": direction,
            "index": candle_index,
            "timestamp": candle["timestamp"],
            "available_timestamp": available_timestamp,
            "level": selected["price"],
            "broken_swing_type": selected["type"],
            "broken_swing_index": selected["index"],
            "broken_swing_timestamp": selected["timestamp"],
            "swing_confirmation_index": selected["confirmation_index"],
            "crossed_levels": crossed_levels,
            "ambiguous_break": False,
        })

        market_bias = direction

    return events


def _latest_event_bias(events, last_available_timestamp):
    valid = [
        event for event in events
        if (
            event.get("direction") in ("BULLISH", "BEARISH")
            and event.get("event") in (
                "BOS_BULLISH",
                "CHOCH_BULLISH",
                "BOS_BEARISH",
                "CHOCH_BEARISH",
            )
            and _is_integer(event.get("available_timestamp"))
            and event["available_timestamp"] <= last_available_timestamp
        )
    ]

    if not valid:
        return None

    latest = max(
        valid,
        key=lambda item: (
            item["available_timestamp"],
            item["index"],
        ),
    )

    return latest["direction"]


def analyze_daily_structure(
    candles,
    timeframe="1D",
    internal_left_bars=2,
    internal_right_bars=2,
):
    candles = validate_candles(candles)
    duration = infer_candle_duration(candles)

    internal_swings = detect_swings(
        candles,
        left_bars=internal_left_bars,
        right_bars=internal_right_bars,
        candle_duration=duration,
    )

    external_swings = detect_swings(
        candles,
        left_bars=5,
        right_bars=5,
        candle_duration=duration,
    )

    internal_structure = classify_market_structure(
        candles,
        internal_swings,
    )

    external_structure = classify_market_structure(
        candles,
        external_swings,
    )

    internal_events = analyze_structure_events(
        candles,
        internal_swings,
        duration,
    )

    external_events = analyze_structure_events(
        candles,
        external_swings,
        duration,
    )

    last_available = candles[-1]["timestamp"] + duration

    internal_event_bias = _latest_event_bias(
        internal_events,
        last_available,
    )

    external_event_bias = _latest_event_bias(
        external_events,
        last_available,
    )

    internal_trend = internal_structure["trend"]
    external_trend = external_structure["trend"]

    if internal_trend == external_trend == "BULLISH":
        combined_trend = "BULLISH"
        allowed_direction = "LONG"
    elif internal_trend == external_trend == "BEARISH":
        combined_trend = "BEARISH"
        allowed_direction = "SHORT"
    else:
        combined_trend = "SIDEWAYS"
        allowed_direction = "NONE"

    internal_result = {
        **internal_structure,
        "swings": internal_swings,
        "events": internal_events,
        "event_bias": internal_event_bias,
    }

    external_result = {
        **external_structure,
        "swings": external_swings,
        "events": external_events,
        "event_bias": external_event_bias,
    }

    return {
        "timeframe": timeframe,
        "candle_duration": duration,
        "candle_count": len(candles),
        "last_timestamp": candles[-1]["timestamp"],
        "last_close": candles[-1]["close"],
        "internal": {
            "swings": internal_swings,
            "structure": internal_structure,
            "events": internal_events,
            "event_bias": internal_event_bias,
        },
        "external": {
            "swings": external_swings,
            "structure": external_structure,
            "events": external_events,
            "event_bias": external_event_bias,
        },
        "internal_structure": internal_result,
        "external_structure": external_result,
        "trend": combined_trend,
        "event_bias": external_event_bias,
        "allowed_direction": allowed_direction,
        "trade_directions_to_evaluate": (
            [allowed_direction] if allowed_direction != "NONE" else []
        ),
        "status": "ANALYZED",
    }
