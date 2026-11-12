
import math
from statistics import median


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


def validate_candles(candles):
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    if len(candles) < 2:
        raise ValueError("At least two candles are required")

    validated = []
    previous_timestamp = None

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError(
                f"Candle at index {index} must be a dictionary"
            )

        for field in ("timestamp", "open", "high", "low", "close"):
            if not _is_number(candle.get(field)):
                raise ValueError(
                    f"Invalid {field} at candle index {index}"
                )

        timestamp = candle["timestamp"]

        if timestamp < 0 or not _is_integer(timestamp):
            raise ValueError(
                f"Invalid timestamp at candle index {index}"
            )

        if any(
            candle[field] <= 0
            for field in ("open", "high", "low", "close")
        ):
            raise ValueError(
                f"Prices must be positive at candle index {index}"
            )

        if (
            candle["high"] < max(
                candle["open"],
                candle["close"],
                candle["low"],
            )
            or candle["low"] > min(
                candle["open"],
                candle["close"],
                candle["high"],
            )
        ):
            raise ValueError(
                f"Invalid OHLC relationship at candle index {index}"
            )

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "Candle timestamps must be strictly increasing"
            )

        normalized = dict(candle)
        normalized["timestamp"] = int(timestamp)

        for field in ("open", "high", "low", "close"):
            normalized[field] = float(candle[field])

        if normalized.get("volume") is not None:
            if (
                not _is_number(normalized["volume"])
                or normalized["volume"] < 0
            ):
                raise ValueError(
                    f"Invalid volume at candle index {index}"
                )

            normalized["volume"] = float(normalized["volume"])

        validated.append(normalized)
        previous_timestamp = timestamp

    return validated


def infer_candle_duration(candles):
    if not isinstance(candles, list) or len(candles) < 2:
        raise ValueError(
            "At least two candles are required to infer duration"
        )

    differences = []

    for index in range(1, len(candles)):
        previous = candles[index - 1]["timestamp"]
        current = candles[index]["timestamp"]

        difference = current - previous

        if difference <= 0:
            raise ValueError(
                "Candle timestamps must be strictly increasing"
            )

        differences.append(difference)

    duration = int(median(differences))

    if duration <= 0:
        raise ValueError("Unable to infer candle duration")

    return duration


def was_swing_broken_before_confirmation(candles, swing):
    pivot_index = swing.get("index")
    confirmation_index = swing.get("confirmation_index")
    level = swing.get("price")
    swing_type = swing.get("type")

    if (
        not _is_integer(pivot_index)
        or not _is_integer(confirmation_index)
    ):
        raise ValueError("Invalid swing indices")

    pivot_index = int(pivot_index)
    confirmation_index = int(confirmation_index)

    if (
        pivot_index < 0
        or confirmation_index <= pivot_index
        or confirmation_index >= len(candles)
    ):
        raise ValueError("Swing indices are out of range")

    if swing_type not in ("HIGH", "LOW"):
        raise ValueError("Invalid swing type")

    if not _is_number(level) or level <= 0:
        raise ValueError("Invalid swing price")

    for index in range(
        pivot_index + 1,
        confirmation_index + 1,
    ):
        close_price = candles[index]["close"]

        if swing_type == "HIGH" and close_price > level:
            return True

        if swing_type == "LOW" and close_price < level:
            return True

    return False


def detect_swings(
    candles,
    left_bars=3,
    right_bars=3,
    candle_duration=None,
):
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    for name, value in (
        ("left_bars", left_bars),
        ("right_bars", right_bars),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer"
            )

    candles = validate_candles(candles)

    if candle_duration is None:
        candle_duration = infer_candle_duration(candles)

    if (
        not _is_integer(candle_duration)
        or candle_duration <= 0
    ):
        raise ValueError("Invalid candle duration")

    candle_duration = int(candle_duration)

    swings = []

    if len(candles) < left_bars + right_bars + 1:
        return swings

    for index in range(
        left_bars,
        len(candles) - right_bars,
    ):
        current = candles[index]

        previous_candles = candles[
            index - left_bars:index
        ]

        next_candles = candles[
            index + 1:index + right_bars + 1
        ]

        is_swing_high = (
            current["high"] >= max(
                candle["high"] for candle in next_candles
            )
            and current["high"] > max(
                candle["high"] for candle in previous_candles
            )
        )

        is_swing_low = (
            current["low"] <= min(
                candle["low"] for candle in next_candles
            )
            and current["low"] < min(
                candle["low"] for candle in previous_candles
            )
        )

        if not is_swing_high and not is_swing_low:
            continue

        confirmation_index = index + right_bars

        confirmation_timestamp = (
            candles[confirmation_index]["timestamp"]
            + candle_duration
        )

        ambiguous = is_swing_high and is_swing_low

        candidates = []

        if is_swing_high:
            candidates.append({
                "index": index,
                "timestamp": current["timestamp"],
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": confirmation_timestamp,
                "ambiguous_same_candle": ambiguous,
                "type": "HIGH",
                "price": current["high"],
            })

        if is_swing_low:
            candidates.append({
                "index": index,
                "timestamp": current["timestamp"],
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": confirmation_timestamp,
                "ambiguous_same_candle": ambiguous,
                "type": "LOW",
                "price": current["low"],
            })

        for swing in candidates:
            if not was_swing_broken_before_confirmation(
                candles,
                swing,
            ):
                swings.append(swing)

    swings.sort(
        key=lambda swing: (
            swing["confirmation_index"],
            swing["index"],
            swing["type"],
        )
    )

    return swings


def _valid_swings(swings, candle_count):
    if not isinstance(swings, list):
        raise ValueError("swings must be a list")

    valid = []

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

        valid.append(swing)

    return valid


def _structure_comparison(swings, swing_type):
    selected = sorted(
        [
            swing for swing in swings
            if swing["type"] == swing_type
        ],
        key=lambda swing: (
            swing["confirmation_index"],
            swing["index"],
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


def _structure_reason(trend, high_comparison, low_comparison):
    if trend == "BULLISH":
        return "Confirmed higher high and higher low structure."

    if trend == "BEARISH":
        return "Confirmed lower high and lower low structure."

    missing = []

    if high_comparison is None:
        missing.append("two confirmed swing highs")

    if low_comparison is None:
        missing.append("two confirmed swing lows")

    if missing:
        return (
            "Insufficient confirmed structure: "
            + " and ".join(missing)
            + "."
        )

    return (
        "Swing structure is mixed or does not confirm "
        "a directional trend."
    )


def classify_market_structure(candles, swings):
    candles = validate_candles(candles)
    valid_swings = _valid_swings(swings, len(candles))

    high_comparison, latest_high = _structure_comparison(
        valid_swings,
        "HIGH",
    )

    low_comparison, latest_low = _structure_comparison(
        valid_swings,
        "LOW",
    )

    if high_comparison == "HH" and low_comparison == "HL":
        trend = "BULLISH"

    elif high_comparison == "LH" and low_comparison == "LL":
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    reason = _structure_reason(
        trend,
        high_comparison,
        low_comparison,
    )

    high_count = sum(
        1 for swing in valid_swings
        if swing["type"] == "HIGH"
    )

    low_count = sum(
        1 for swing in valid_swings
        if swing["type"] == "LOW"
    )

    return {
        "trend": trend,
        "reason": reason,
        "high_structure": high_comparison,
        "low_structure": low_comparison,
        "latest_swing_high": latest_high,
        "latest_swing_low": latest_low,
        "swing_high_count": high_count,
        "swing_low_count": low_count,
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

    if (
        not _is_integer(candle_duration)
        or candle_duration <= 0
    ):
        raise ValueError("Invalid candle duration")

    candle_duration = int(candle_duration)

    valid_swings = _valid_swings(swings, len(candles))

    events = []
    broken_swings = set()
    market_bias = None

    for candle_index, candle in enumerate(candles):
        available_swings = [
            swing for swing in valid_swings
            if (
                swing["confirmation_index"] < candle_index
                and _swing_identity(swing) not in broken_swings
            )
        ]

        candidates = []

        for swing in available_swings:
            if (
                swing["type"] == "HIGH"
                and candle["close"] > swing["price"]
            ):
                direction = "BULLISH"

            elif (
                swing["type"] == "LOW"
                and candle["close"] < swing["price"]
            ):
                direction = "BEARISH"

            else:
                continue

            candidates.append((swing, direction))

        if not candidates:
            continue

        for swing, direction in candidates:
            broken_swings.add(_swing_identity(swing))

        crossed_directions = {
            direction for _, direction in candidates
        }

        candidates.sort(
            key=lambda item: (
                item[0]["confirmation_index"],
                item[0]["index"],
            ),
            reverse=True,
        )

        crossed_levels = [
            {
                "swing_type": swing["type"],
                "direction": direction,
                "level": swing["price"],
                "swing_index": swing["index"],
                "swing_timestamp": swing["timestamp"],
                "swing_confirmation_index": swing[
                    "confirmation_index"
                ],
                "swing_confirmation_timestamp": swing.get(
                    "confirmation_timestamp"
                ),
            }
            for swing, direction in candidates
        ]

        event_timestamp = candle["timestamp"]
        available_timestamp = (
            event_timestamp + candle_duration
        )

        if len(crossed_directions) > 1:
            events.append({
                "event_type": "AMBIGUOUS",
                "type": "AMBIGUOUS",
                "direction": "UNKNOWN",
                "event": "AMBIGUOUS_BREAK",
                "structure_event": "AMBIGUOUS_BREAK",
                "level": None,
                "price": None,
                "level_timestamp": None,
                "index": candle_index,
                "timestamp": event_timestamp,
                "available_timestamp": available_timestamp,
                "confirmation_index": candle_index,
                "confirmation_timestamp": available_timestamp,
                "broken_swing_type": "MULTIPLE",
                "broken_swing_index": None,
                "broken_swing_timestamp": None,
                "swing_confirmation_index": None,
                "swing_confirmation_timestamp": None,
                "previous_bias": market_bias or "NEUTRAL",
                "new_bias": market_bias or "NEUTRAL",
                "crossed_levels": crossed_levels,
                "crossed_level_count": len(crossed_levels),
                "ambiguous_break": True,
            })

            continue

        selected_swing, direction = candidates[0]

        event_type = (
            "BOS"
            if market_bias is None or direction == market_bias
            else "CHOCH"
        )

        events.append({
            "event_type": event_type,
            "type": event_type,
            "direction": direction,
            "event": f"{event_type}_{direction}",
            "structure_event": f"{event_type}_{direction}",
            "level": selected_swing["price"],
            "price": selected_swing["price"],
            "level_timestamp": selected_swing["timestamp"],
            "index": candle_index,
            "timestamp": event_timestamp,
            "available_timestamp": available_timestamp,
            "confirmation_index": candle_index,
            "confirmation_timestamp": available_timestamp,
            "broken_swing_type": selected_swing["type"],
            "broken_swing_index": selected_swing["index"],
            "broken_swing_timestamp": selected_swing["timestamp"],
            "swing_confirmation_index": selected_swing[
                "confirmation_index"
            ],
            "swing_confirmation_timestamp": selected_swing.get(
                "confirmation_timestamp"
            ),
            "previous_bias": market_bias or "NEUTRAL",
            "new_bias": direction,
            "crossed_levels": crossed_levels,
            "crossed_level_count": len(crossed_levels),
            "ambiguous_break": False,
        })

        market_bias = direction

    events.sort(
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
            event["event"],
        )
    )

    return events


def _latest_event_bias(events, last_available_timestamp):
    valid_events = [
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

    if not valid_events:
        return None

    latest = max(
        valid_events,
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
        ),
    )

    return latest["direction"]


def analyze_daily_structure(
    candles,
    timeframe="1D",
    internal_left_bars=2,
    internal_right_bars=2,
):
    if (
        isinstance(internal_left_bars, bool)
        or not isinstance(internal_left_bars, int)
        or internal_left_bars < 1
    ):
        raise ValueError(
            "internal_left_bars must be a positive integer"
        )

    if (
        isinstance(internal_right_bars, bool)
        or not isinstance(internal_right_bars, int)
        or internal_right_bars < 1
    ):
        raise ValueError(
            "internal_right_bars must be a positive integer"
        )

    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string")

    candles = validate_candles(candles)
    candle_duration = infer_candle_duration(candles)

    internal_swings = detect_swings(
        candles,
        left_bars=internal_left_bars,
        right_bars=internal_right_bars,
        candle_duration=candle_duration,
    )

    external_swings = detect_swings(
        candles,
        left_bars=5,
        right_bars=5,
        candle_duration=candle_duration,
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
        candle_duration,
    )

    external_events = analyze_structure_events(
        candles,
        external_swings,
        candle_duration,
    )

    last_available_timestamp = (
        candles[-1]["timestamp"] + candle_duration
    )

    internal_event_bias = _latest_event_bias(
        internal_events,
        last_available_timestamp,
    )

    external_event_bias = _latest_event_bias(
        external_events,
        last_available_timestamp,
    )

    internal_trend = internal_structure["trend"]
    external_trend = external_structure["trend"]

    if (
        internal_trend == "BULLISH"
        and external_trend == "BULLISH"
    ):
        combined_trend = "BULLISH"

    elif (
        internal_trend == "BEARISH"
        and external_trend == "BEARISH"
    ):
        combined_trend = "BEARISH"

    else:
        combined_trend = "SIDEWAYS"

    if combined_trend == "BULLISH":
        allowed_direction = "LONG"
        trade_directions = ["LONG"]

    elif combined_trend == "BEARISH":
        allowed_direction = "SHORT"
        trade_directions = ["SHORT"]

    else:
        allowed_direction = "NONE"
        trade_directions = []

    external_reason = external_structure["reason"]

    return {
        "timeframe": timeframe,
        "candle_duration": candle_duration,
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

        "internal_structure": {
            **internal_structure,
            "swings": internal_swings,
            "events": internal_events,
            "event_bias": internal_event_bias,
        },

        "external_structure": {
            **external_structure,
            "swings": external_swings,
            "events": external_events,
            "event_bias": external_event_bias,
            "reason": external_reason,
        },

        "context": {
            "internal_trend": internal_trend,
            "external_trend": external_trend,
            "internal_event_bias": internal_event_bias,
            "external_event_bias": external_event_bias,
        },

        "trend": combined_trend,
        "event_bias": external_event_bias,
        "allowed_direction": allowed_direction,
        "trade_directions_to_evaluate": trade_directions,
        "status": "ANALYZED",
    }
