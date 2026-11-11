
import math


def validate_candles(candles):
    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    if len(candles) < 20:
        raise ValueError("At least 20 candles are required")

    required_fields = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    )

    validated = []
    previous_timestamp = None

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError(
                f"Candle at index {index} must be a dictionary"
            )

        for field in required_fields:
            if field not in candle:
                raise ValueError(
                    f"Missing field '{field}' at candle index {index}"
                )

        converted = dict(candle)

        for field in required_fields:
            value = converted[field]

            if isinstance(value, bool):
                raise ValueError(
                    f"Invalid {field} at candle index {index}"
                )

            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid {field} at candle index {index}"
                )

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite {field} at candle index {index}"
                )

            converted[field] = value

        if converted["high"] < converted["low"]:
            raise ValueError(
                f"Invalid high/low at candle index {index}"
            )

        if converted["high"] < max(
            converted["open"],
            converted["close"],
        ):
            raise ValueError(
                f"Invalid high at candle index {index}"
            )

        if converted["low"] > min(
            converted["open"],
            converted["close"],
        ):
            raise ValueError(
                f"Invalid low at candle index {index}"
            )

        timestamp = converted["timestamp"]

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "Candle timestamps must be strictly increasing"
            )

        previous_timestamp = timestamp
        validated.append(converted)

    return validated


def infer_candle_duration(candles):
    if len(candles) < 2:
        raise ValueError(
            "At least two candles are required"
        )

    differences = []

    for index in range(1, len(candles)):
        difference = (
            candles[index]["timestamp"]
            - candles[index - 1]["timestamp"]
        )

        if difference > 0:
            differences.append(difference)

    if not differences:
        raise ValueError(
            "Unable to infer candle duration"
        )

    differences.sort()

    return differences[len(differences) // 2]


def was_swing_broken_before_confirmation(candles, swing):
    pivot_index = swing["index"]
    confirmation_index = swing["confirmation_index"]
    level = swing["price"]
    swing_type = swing["type"]

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
):
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
            candle["high"]
            for candle in previous_candles
        ]

        next_highs = [
            candle["high"]
            for candle in next_candles
        ]

        previous_lows = [
            candle["low"]
            for candle in previous_candles
        ]

        next_lows = [
            candle["low"]
            for candle in next_candles
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

        if is_swing_high:
            swing = {
                "index": index,
                "timestamp": current["timestamp"],
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": (
                    candles[confirmation_index]["timestamp"]
                ),
                "ambiguous_same_candle": is_swing_low,
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
                "confirmation_timestamp": (
                    candles[confirmation_index]["timestamp"]
                ),
                "ambiguous_same_candle": is_swing_high,
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


def classify_market_structure(swings):
    highs = [
        swing for swing in swings
        if swing["type"] == "HIGH"
    ]

    lows = [
        swing for swing in swings
        if swing["type"] == "LOW"
    ]

    higher_highs = 0
    lower_highs = 0
    higher_lows = 0
    lower_lows = 0

    for previous, current in zip(highs, highs[1:]):
        if current["price"] > previous["price"]:
            higher_highs += 1
        elif current["price"] < previous["price"]:
            lower_highs += 1

    for previous, current in zip(lows, lows[1:]):
        if current["price"] > previous["price"]:
            higher_lows += 1
        elif current["price"] < previous["price"]:
            lower_lows += 1

    if (
        higher_highs > lower_highs
        and higher_lows > lower_lows
    ):
        trend = "BULLISH"

    elif (
        lower_highs > higher_highs
        and lower_lows > higher_lows
    ):
        trend = "BEARISH"

    else:
        trend = "SIDEWAYS"

    return {
        "trend": trend,
        "higher_highs": higher_highs,
        "lower_highs": lower_highs,
        "higher_lows": higher_lows,
        "lower_lows": lower_lows,
        "swing_high_count": len(highs),
        "swing_low_count": len(lows),
    }


def analyze_structure_events(
    candles,
    swings,
    candle_duration,
):
    events = []

    active_highs = []
    active_lows = []

    broken_levels = set()

    previous_direction = 0

    swings_by_confirmation = {}

    for swing in swings:
        confirmation_index = swing["confirmation_index"]

        swings_by_confirmation.setdefault(
            confirmation_index,
            [],
        ).append(swing)

    for candle_index, candle in enumerate(candles):
        newly_confirmed = swings_by_confirmation.get(
            candle_index,
            [],
        )

        for swing in newly_confirmed:
            if swing["type"] == "HIGH":
                active_highs.append(swing)

            elif swing["type"] == "LOW":
                active_lows.append(swing)

        if not newly_confirmed:
            pass

        close_price = candle["close"]

        broken_candidates = []

        for swing in active_highs:
            key = ("HIGH", swing["index"])

            if key in broken_levels:
                continue

            if candle_index <= swing["confirmation_index"]:
                continue

            if close_price > swing["price"]:
                broken_candidates.append(
                    (swing, 1, key)
                )

        for swing in active_lows:
            key = ("LOW", swing["index"])

            if key in broken_levels:
                continue

            if candle_index <= swing["confirmation_index"]:
                continue

            if close_price < swing["price"]:
                broken_candidates.append(
                    (swing, -1, key)
                )

        if not broken_candidates:
            continue

        broken_candidates.sort(
            key=lambda item: (
                item[0]["confirmation_index"],
                item[0]["index"],
            ),
            reverse=True,
        )

        selected_swing, direction, selected_key = (
            broken_candidates[0]
        )

        if previous_direction == 0:
            event_name = (
                "BOS_BULLISH"
                if direction == 1
                else "BOS_BEARISH"
            )

        elif previous_direction == direction:
            event_name = (
                "BOS_BULLISH"
                if direction == 1
                else "BOS_BEARISH"
            )

        else:
            event_name = (
                "CHOCH_BULLISH"
                if direction == 1
                else "CHOCH_BEARISH"
            )

        events.append({
            "event": event_name,
            "type": event_name,
            "direction": (
                "BULLISH"
                if direction == 1
                else "BEARISH"
            ),
            "level": selected_swing["price"],
            "level_timestamp": selected_swing["timestamp"],
            "timestamp": candle["timestamp"],
            "index": candle_index,
            "swing_index": selected_swing["index"],
            "confirmation_timestamp": (
                selected_swing["confirmation_timestamp"]
            ),
            "available_timestamp": (
                candle["timestamp"] + candle_duration
            ),
        })

        broken_levels.add(selected_key)

        previous_direction = direction

    return events


def analyze_daily_structure(
    candles,
    timeframe="1D",
    internal_left_bars=2,
    internal_right_bars=2,
):
    candles = validate_candles(candles)

    for name, value in (
        ("internal_left_bars", internal_left_bars),
        ("internal_right_bars", internal_right_bars),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"{name} must be a positive integer"
            )

    candle_duration = infer_candle_duration(candles)

    internal_swings = detect_swings(
        candles,
        left_bars=internal_left_bars,
        right_bars=internal_right_bars,
    )

    external_swings = detect_swings(
        candles,
        left_bars=5,
        right_bars=5,
    )

    internal_structure = classify_market_structure(
        internal_swings
    )

    external_structure = classify_market_structure(
        external_swings
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

    internal_data = {
        "swings": internal_swings,
        "structure": internal_structure,
        "events": internal_events,
    }

    external_data = {
        "swings": external_swings,
        "structure": external_structure,
        "events": external_events,
    }

    return {
        "timeframe": timeframe,
        "internal_structure": internal_data,
        "external_structure": external_data,
        "internal": internal_data,
        "external": external_data,
        "context": {
            "internal_trend": internal_structure.get("trend"),
            "external_trend": external_structure.get("trend"),
        },
        "trend": external_structure.get("trend"),
        "allowed_direction": "BOTH",
        "trade_directions_to_evaluate": [
            "LONG",
            "SHORT",
        ],
        "status": "ANALYZED",
    }
