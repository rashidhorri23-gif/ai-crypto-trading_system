
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

    if (
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not math.isfinite(level)
        or level <= 0
    ):
        raise ValueError("Invalid swing price")

    for index in range(
        pivot_index + 1,
        confirmation_index + 1,
    ):
        candle = candles[index]

        if not isinstance(candle, dict):
            raise ValueError(
                f"Invalid candle at index {index}"
            )

        close_price = candle.get("close")

        if (
            isinstance(close_price, bool)
            or not isinstance(close_price, (int, float))
            or not math.isfinite(close_price)
            or close_price <= 0
        ):
            raise ValueError(
                f"Invalid close price at index {index}"
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

    if len(candles) < 2:
        raise ValueError(
            "At least two candles are required"
        )

    if candle_duration is None:
        candle_duration = infer_candle_duration(candles)

    if (
        isinstance(candle_duration, bool)
        or not isinstance(candle_duration, (int, float))
        or not math.isfinite(candle_duration)
        or candle_duration <= 0
        or int(candle_duration) != candle_duration
    ):
        raise ValueError("Invalid candle duration")

    candle_duration = int(candle_duration)

    required_fields = (
        "timestamp",
        "high",
        "low",
        "close",
    )

    previous_timestamp = None

    for index, candle in enumerate(candles):
        if not isinstance(candle, dict):
            raise ValueError(
                f"Candle at index {index} must be a dictionary"
            )

        for field in required_fields:
            value = candle.get(field)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Invalid {field} at candle index {index}"
                )

        if (
            candle["high"] <= 0
            or candle["low"] <= 0
            or candle["close"] <= 0
            or candle["high"] < candle["low"]
        ):
            raise ValueError(
                f"Invalid OHLC values at candle index {index}"
            )

        timestamp = candle["timestamp"]

        if (
            timestamp < 0
            or int(timestamp) != timestamp
        ):
            raise ValueError(
                f"Invalid timestamp at candle index {index}"
            )

        if (
            previous_timestamp is not None
            and timestamp <= previous_timestamp
        ):
            raise ValueError(
                "Candle timestamps must be strictly increasing"
            )

        previous_timestamp = timestamp

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
