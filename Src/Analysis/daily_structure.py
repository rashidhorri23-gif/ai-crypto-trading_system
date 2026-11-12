
import math
from statistics import median


def _is_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
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

        required_fields = (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        )

        for field in required_fields:
            value = candle.get(field)

            if not _is_number(value):
                raise ValueError(
                    f"Invalid {field} at candle index {index}"
                )

        timestamp = candle["timestamp"]

        if timestamp < 0 or int(timestamp) != timestamp:
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

        if candle["high"] < candle["low"]:
            raise ValueError(
                f"High is below low at candle index {index}"
            )

        if (
            candle["high"] < candle["open"]
            or candle["high"] < candle["close"]
            or candle["low"] > candle["open"]
            or candle["low"] > candle["close"]
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

        if "volume" in normalized and normalized["volume"] is not None:
            if not _is_number(normalized["volume"]):
                raise ValueError(
                    f"Invalid volume at candle index {index}"
                )

        validated.append(normalized)
        previous_timestamp = timestamp

    return validated


def infer_candle_duration(candles):
    if not isinstance(candles, list) or len(candles) < 2:
        raise ValueError(
            "At least two candles are required to infer duration"
        )

    differences = []
    previous_timestamp = None

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError(
                f"Invalid candle at index {index}"
            )

        timestamp = candle.get("timestamp")

        if not _is_number(timestamp):
            raise ValueError(
                f"Invalid timestamp at index {index}"
            )

        if timestamp < 0 or int(timestamp) != timestamp:
            raise ValueError(
                f"Invalid timestamp at index {index}"
            )

        timestamp = int(timestamp)

        if previous_timestamp is not None:
            difference = timestamp - previous_timestamp

            if difference <= 0:
                raise ValueError(
                    "Candle timestamps must be strictly increasing"
                )

            differences.append(difference)

        previous_timestamp = timestamp

    duration = int(median(differences))

    if duration <= 0:
        raise ValueError("Unable to infer candle duration")

    return duration


def was_swing_broken_before_confirmation(candles, swing):
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    if not isinstance(swing, dict):
        raise ValueError("swing must be a dictionary")

    pivot_index = swing.get("index")
    confirmation_index = swing.get("confirmation_index")
    level = swing.get("price")
    swing_type = swing.get("type")

    if (
        isinstance(pivot_index, bool)
        or not isinstance(pivot_index, int)
        or isinstance(confirmation_index, bool)
        or not isinstance(confirmation_index, int)
    ):
        raise ValueError("Invalid swing indices")

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
        close_price = candles[index].get("close")

        if not _is_number(close_price) or close_price <= 0:
            raise ValueError(
                f"Invalid close price at candle index {index}"
            )

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
        not _is_number(candle_duration)
        or candle_duration <= 0
        or int(candle_duration) != candle_duration
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

        previous_highs = [
            candle["high"] for candle in previous_candles
        ]

        next_highs = [
            candle["high"] for candle in next_candles
        ]

        previous_lows = [
            candle["low"] for candle in previous_candles
        ]

        next_lows = [
            candle["low"] for candle in next_candles
        ]

        is_swing_high = (
            current["high"] > max(
                previous_highs + next_highs
            )
        )

        is_swing_low = (
            current["low"] < min(
                previous_lows + next_lows
            )
        )

        if not is_swing_high and not is_swing_low:
            continue

        confirmation_index = index + right_bars

        confirmation_timestamp = (
            candles[confirmation_index]["timestamp"]
            + candle_duration
        )

        ambiguous_same_candle = (
            is_swing_high and is_swing_low
        )

        if is_swing_high:
            swing = {
                "index": index,
                "timestamp": current["timestamp"],
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": confirmation_timestamp,
                "ambiguous_same_candle": ambiguous_same_candle,
                "type": "HIGH",
                "price": current["high"],
            }

            if not was_swing_broken_before_confirmation(
                candles,
                swing,
            ):
                swings.append(swing)

        if is_swing_low:
            swing = {
                "index": index,
                "timestamp": current["timestamp"],
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": confirmation_timestamp,
                "ambiguous_same_candle": ambiguous_same_candle,
                "type": "LOW",
                "price": current["low"],
            }

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


def classify_market_structure(candles, swings):
    candles = validate_candles(candles)

    if not isinstance(swings, list):
        raise ValueError("swings must be a list")

    highs = sorted(
        [
            swing for swing in swings
            if isinstance(swing, dict)
            and swing.get("type") == "HIGH"
        ],
        key=lambda swing: swing.get("confirmation_index", -1),
    )

    lows = sorted(
        [
            swing for swing in swings
            if isinstance(swing, dict)
            and swing.get("type") == "LOW"
        ],
        key=lambda swing: swing.get("confirmation_index", -1),
    )

    high_comparison = None
    low_comparison = None

    if len(highs) >= 2:
        previous_high = highs[-2]["price"]
        latest_high = highs[-1]["price"]

        if latest_high > previous_high:
            high_comparison = "HH"
        elif latest_high < previous_high:
            high_comparison = "LH"
        else:
            high_comparison = "EQH"

    if len(lows) >= 2:
        previous_low = lows[-2]["price"]
        latest_low = lows[-1]["price"]

        if latest_low > previous_low:
            low_comparison = "HL"
        elif latest_low < previous_low:
            low_comparison = "LL"
        else:
            low_comparison = "EQL"

    if high_comparison == "HH" and low_comparison == "HL":
        trend = "BULLISH"
    elif high_comparison == "LH" and low_comparison == "LL":
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    latest_high = highs[-1] if highs else None
    latest_low = lows[-1] if lows else None

    return {
        "trend": trend,
        "high_structure": high_comparison,
        "low_structure": low_comparison,
        "latest_swing_high": latest_high,
        "latest_swing_low": latest_low,
        "swing_high_count": len(highs),
        "swing_low_count": len(lows),
        "last_close": candles[-1]["close"],
        "last_timestamp": candles[-1]["timestamp"],
    }


def analyze_structure_events(candles, swings, candle_duration):
    candles = validate_candles(candles)

    if not isinstance(swings, list):
        raise ValueError("swings must be a list")

    if (
        not _is_number(candle_duration)
        or candle_duration <= 0
        or int(candle_duration) != candle_duration
    ):
        raise ValueError("Invalid candle duration")

    candle_duration = int(candle_duration)

    valid_swings = []

    for swing in swings:
        if not isinstance(swing, dict):
            continue

        swing_type = swing.get("type")
        index = swing.get("index")
        confirmation_index = swing.get("confirmation_index")
        price = swing.get("price")

        if swing_type not in ("HIGH", "LOW"):
            continue

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or isinstance(confirmation_index, bool)
            or not isinstance(confirmation_index, int)
        ):
            continue

        if (
            index < 0
            or confirmation_index <= index
            or confirmation_index >= len(candles)
        ):
            continue

        if not _is_number(price) or price <= 0:
            continue

        valid_swings.append(swing)

    valid_swings.sort(
        key=lambda swing: (
            swing["confirmation_index"],
            swing["index"],
        )
    )

    events = []
    broken_swings = set()
    market_bias = None

    for candle_index, candle in enumerate(candles):
        available_swings = [
            swing for swing in valid_swings
            if (
                swing["confirmation_index"] < candle_index
                and id(swing) not in broken_swings
            )
        ]

        candidates = []

        for swing in available_swings:
            if swing.get("ambiguous_same_candle"):
                continue

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

        candidates.sort(
            key=lambda item: (
                item[0]["confirmation_index"],
                item[0]["index"],
            ),
            reverse=True,
        )

        selected_swing, direction = candidates[0]

        if market_bias is None or direction == market_bias:
            event_type = "BOS"
        else:
            event_type = "CHOCH"

        event_timestamp = candle["timestamp"]
        available_timestamp = (
            event_timestamp + candle_duration
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
        })

        # All levels crossed in this candle are consumed.
        # Only the most recently confirmed eligible level creates an event.
        for swing, candidate_direction in candidates:
            broken_swings.add(id(swing))

        market_bias = direction

    events.sort(
        key=lambda event: (
            event["available_timestamp"],
            event["index"],
            event["event"],
        )
    )

    return events


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
        },
        "external": {
            "swings": external_swings,
            "structure": external_structure,
            "events": external_events,
        },
        "internal_structure": {
            **internal_structure,
            "swings": internal_swings,
            "events": internal_events,
        },
        "external_structure": {
            **external_structure,
            "swings": external_swings,
            "events": external_events,
        },
        "context": {
            "internal_trend": internal_trend,
            "external_trend": external_trend,
        },
        "trend": combined_trend,
        "allowed_direction": "BOTH",
        "trade_directions_to_evaluate": [
            "LONG",
            "SHORT",
        ],
        "status": "ANALYZED",
    }
