
from typing import Dict, List, Any
from math import isfinite
from statistics import median


REQUIRED_FIELDS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
    "2H": 7200,
    "4H": 14400,
    "6H": 21600,
    "12H": 43200,
    "1D": 86400,
    "1W": 604800,
}

VALID_SWING_STATUSES = {
    "ACTIVE",
    "BROKEN",
    "PRE_BROKEN",
    "CONFIRMED_PENDING_NEXT_CANDLE",
}


def validate_candles(candles: List[Dict[str, Any]]) -> None:
    if not candles or len(candles) < 20:
        raise ValueError("At least 20 candles are required.")

    previous_timestamp = None

    for index, candle in enumerate(candles):
        for field in REQUIRED_FIELDS:
            if field not in candle:
                raise ValueError(f"Missing field: {field}")

        try:
            timestamp = int(float(candle["timestamp"]))

            values = {
                field: float(candle[field])
                for field in REQUIRED_FIELDS[1:]
            }
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid numeric candle value at index {index}."
            )

        if timestamp <= 0:
            raise ValueError(
                f"Invalid timestamp at index {index}."
            )

        if previous_timestamp is not None:
            if timestamp <= previous_timestamp:
                raise ValueError(
                    "Candles must be uniquely sorted by timestamp."
                )

        if not all(isfinite(value) for value in values.values()):
            raise ValueError(
                f"Non-finite candle value at index {index}."
            )

        open_price = values["open"]
        high = values["high"]
        low = values["low"]
        close = values["close"]
        volume = values["volume"]

        if min(open_price, high, low, close) <= 0:
            raise ValueError(
                f"Prices must be positive at index {index}."
            )

        if volume < 0:
            raise ValueError(
                f"Negative volume at index {index}."
            )

        if high < max(open_price, close, low):
            raise ValueError(
                f"Invalid candle high at index {index}."
            )

        if low > min(open_price, close, high):
            raise ValueError(
                f"Invalid candle low at index {index}."
            )

        previous_timestamp = timestamp


def infer_candle_duration(
    candles: List[Dict[str, Any]],
    timeframe: str
) -> int:

    if timeframe in TIMEFRAME_SECONDS:
        return TIMEFRAME_SECONDS[timeframe]

    timestamps = [
        int(float(candle["timestamp"]))
        for candle in candles
    ]

    differences = [
        timestamps[i] - timestamps[i - 1]
        for i in range(1, len(timestamps))
        if timestamps[i] > timestamps[i - 1]
    ]

    if not differences:
        raise ValueError("Cannot infer candle duration.")

    return int(median(differences))


def was_swing_broken_before_confirmation(
    candles: List[Dict[str, Any]],
    swing: Dict[str, Any]
) -> bool:

    pivot_index = int(swing["index"])
    confirmation_index = int(swing["confirmation_index"])
    level = float(swing["price"])
    swing_type = str(swing["type"]).upper()

    if (
        pivot_index < 0
        or confirmation_index >= len(candles)
        or confirmation_index <= pivot_index
    ):
        raise ValueError("Invalid swing confirmation indexes.")

    for index in range(
        pivot_index + 1,
        confirmation_index + 1
    ):
        close = float(candles[index]["close"])

        if swing_type == "HIGH" and close > level:
            return True

        if swing_type == "LOW" and close < level:
            return True

    return False


def detect_swings(
    candles: List[Dict[str, Any]],
    left_bars: int = 3,
    right_bars: int = 3
) -> List[Dict[str, Any]]:

    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing windows must be positive.")

    swings = []

    for i in range(
        left_bars,
        len(candles) - right_bars
    ):
        high = float(candles[i]["high"])
        low = float(candles[i]["low"])

        previous_highs = [
            float(candles[j]["high"])
            for j in range(i - left_bars, i)
        ]

        next_highs = [
            float(candles[j]["high"])
            for j in range(i + 1, i + right_bars + 1)
        ]

        previous_lows = [
            float(candles[j]["low"])
            for j in range(i - left_bars, i)
        ]

        next_lows = [
            float(candles[j]["low"])
            for j in range(i + 1, i + right_bars + 1)
        ]

        confirmation_index = i + right_bars

        is_swing_high = (
            high > max(previous_highs + next_highs)
        )

        is_swing_low = (
            low < min(previous_lows + next_lows)
        )

        ambiguous_same_candle = (
            is_swing_high and is_swing_low
        )

        common_fields = {
            "index": i,
            "timestamp": int(
                float(candles[i]["timestamp"])
            ),
            "confirmation_index": confirmation_index,
            "confirmation_timestamp": int(
                float(
                    candles[confirmation_index]["timestamp"]
                )
            ),
            "ambiguous_same_candle": ambiguous_same_candle,
        }

        if is_swing_high:
            swings.append({
                **common_fields,
                "type": "HIGH",
                "price": high,
            })

        if is_swing_low:
            swings.append({
                **common_fields,
                "type": "LOW",
                "price": low,
            })

    return sorted(
        swings,
        key=lambda item: (
            item["confirmation_index"],
            item["index"],
            item["type"]
        )
    )


def classify_market_structure(
    swings: List[Dict[str, Any]],
    structure_type: str
) -> Dict[str, Any]:

    highs = [
        swing for swing in swings
        if swing["type"] == "HIGH"
    ]

    lows = [
        swing for swing in swings
        if swing["type"] == "LOW"
    ]

    ambiguous_indices = sorted({
        swing["index"]
        for swing in swings
        if swing.get("ambiguous_same_candle", False)
    })

    result = {
        "structure_type": structure_type,
        "trend": "UNKNOWN",
        "reason": "Insufficient confirmed swing points.",
        "high_count": len(highs),
        "low_count": len(lows),
        "ambiguous_pivot_indices": ambiguous_indices,
    }

    if len(highs) < 2 or len(lows) < 2:
        return result

    previous_high = highs[-2]
    latest_high = highs[-1]

    previous_low = lows[-2]
    latest_low = lows[-1]

    higher_high = (
        latest_high["price"] > previous_high["price"]
    )

    higher_low = (
        latest_low["price"] > previous_low["price"]
    )

    lower_high = (
        latest_high["price"] < previous_high["price"]
    )

    lower_low = (
        latest_low["price"] < previous_low["price"]
    )

    if higher_high and higher_low:
        trend = "BULLISH"
        reason = "Higher High and Higher Low."

    elif lower_high and lower_low:
        trend = "BEARISH"
        reason = "Lower High and Lower Low."

    else:
        trend = "RANGE"
        reason = "Swing structure is mixed or ranging."

    return {
        "structure_type": structure_type,
        "trend": trend,
        "reason": reason,
        "high_count": len(highs),
        "low_count": len(lows),
        "previous_high": previous_high,
        "latest_high": latest_high,
        "previous_low": previous_low,
        "latest_low": latest_low,
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
        "ambiguous_pivot_indices": ambiguous_indices,
    }


def analyze_structure_events(
    candles: List[Dict[str, Any]],
    swings: List[Dict[str, Any]],
    structure_type: str,
    candle_duration: int
) -> Dict[str, Any]:

    swings_by_confirmation = {}
    tracked_swings = []

    for original_swing in swings:
        swing = dict(original_swing)
        swing["status"] = "PENDING"

        confirmation_index = swing["confirmation_index"]

        if confirmation_index + 1 < len(candles):
            swing["available_timestamp"] = int(
                float(candles[confirmation_index + 1]["timestamp"])
            )
        else:
            swing["available_timestamp"] = (
                swing["confirmation_timestamp"] + candle_duration
            )

        tracked_swings.append(swing)

        swings_by_confirmation.setdefault(
            confirmation_index, []
        ).append(swing)

    active_highs = []
    active_lows = []

    events = []
    bias = "UNKNOWN"

    for i, candle in enumerate(candles):

        previous_close = None

        if i > 0:
            previous_close = float(
                candles[i - 1]["close"]
            )

        current_close = float(candle["close"])

        current_timestamp = int(
            float(candle["timestamp"])
        )

        newly_confirmed = swings_by_confirmation.get(
            i - 1, []
        )

        for swing in newly_confirmed:

            if was_swing_broken_before_confirmation(
                candles,
                swing
            ):
                swing["status"] = "PRE_BROKEN"
                continue

            if swing["type"] == "HIGH":
                swing["status"] = "ACTIVE"
                active_highs.append(swing)

            else:
                swing["status"] = "ACTIVE"
                active_lows.append(swing)

        current_structure = classify_market_structure(
            active_highs + active_lows,
            structure_type
        )

        if bias == "UNKNOWN":
            if current_structure["trend"] in (
                "BULLISH",
                "BEARISH"
            ):
                bias = current_structure["trend"]

        if i == 0:
            continue

        bullish_candidates = [
            swing for swing in active_highs
            if (
                swing["status"] == "ACTIVE"
                and previous_close <= swing["price"]
                and current_close > swing["price"]
            )
        ]

        bearish_candidates = [
            swing for swing in active_lows
            if (
                swing["status"] == "ACTIVE"
                and previous_close >= swing["price"]
                and current_close < swing["price"]
            )
        ]

        if bullish_candidates and bearish_candidates:

            events.append({
                "event": "AMBIGUOUS_DOUBLE_BREAK",
                "structure_type": structure_type,
                "index": i,
                "timestamp": current_timestamp,
                "candle_close_timestamp":
                    current_timestamp + candle_duration,
                "available_timestamp":
                    current_timestamp + candle_duration,
                "close": current_close,
                "bullish_levels": [
                    swing["price"]
                    for swing in bullish_candidates
                ],
                "bearish_levels": [
                    swing["price"]
                    for swing in bearish_candidates
                ],
                "confirmation": "CANDLE_CLOSE",
                "reason": (
                    "Conflicting structural boundaries were crossed."
                ),
            })

            for swing in bullish_candidates + bearish_candidates:
                swing["status"] = "BROKEN"

            bias = "UNKNOWN"
            continue

        if bullish_candidates:

            bullish_candidates.sort(
                key=lambda item: item["price"]
            )

            selected_swing = bullish_candidates[-1]

            event_type = (
                "CHOCH_BULLISH"
                if bias == "BEARISH"
                else "BOS_BULLISH"
            )

            events.append({
                "event": event_type,
                "structure_type": structure_type,
                "index": i,
                "timestamp": current_timestamp,
                "candle_close_timestamp":
                    current_timestamp + candle_duration,
                "available_timestamp":
                    current_timestamp + candle_duration,
                "level": selected_swing["price"],
                "level_timestamp": selected_swing["timestamp"],
                "level_confirmation_timestamp":
                    selected_swing["confirmation_timestamp"],
                "level_available_timestamp":
                    selected_swing["available_timestamp"],
                "crossed_levels": [
                    {
                        "price": swing["price"],
                        "timestamp": swing["timestamp"],
                        "type": swing["type"],
                    }
                    for swing in bullish_candidates
                ],
                "previous_close": previous_close,
                "close": current_close,
                "confirmation": "CANDLE_CLOSE",
            })

            for swing in bullish_candidates:
                swing["status"] = "BROKEN"

            bias = "BULLISH"

        elif bearish_candidates:

            bearish_candidates.sort(
                key=lambda item: item["price"]
            )

            selected_swing = bearish_candidates[0]

            event_type = (
                "CHOCH_BEARISH"
                if bias == "BULLISH"
                else "BOS_BEARISH"
            )

            events.append({
                "event": event_type,
                "structure_type": structure_type,
                "index": i,
                "timestamp": current_timestamp,
                "candle_close_timestamp":
                    current_timestamp + candle_duration,
                "available_timestamp":
                    current_timestamp + candle_duration,
                "level": selected_swing["price"],
                "level_timestamp": selected_swing["timestamp"],
                "level_confirmation_timestamp":
                    selected_swing["confirmation_timestamp"],
                "level_available_timestamp":
                    selected_swing["available_timestamp"],
                "crossed_levels": [
                    {
                        "price": swing["price"],
                        "timestamp": swing["timestamp"],
                        "type": swing["type"],
                    }
                    for swing in bearish_candidates
                ],
                "previous_close": previous_close,
                "close": current_close,
                "confirmation": "CANDLE_CLOSE",
            })

            for swing in bearish_candidates:
                swing["status"] = "BROKEN"

            bias = "BEARISH"

    final_confirmed_swings = [
        swing for swing in tracked_swings
        if swing["confirmation_index"] <= len(candles) - 1
    ]

    for swing in final_confirmed_swings:
        if swing["status"] == "PENDING":
            swing["status"] = (
                "CONFIRMED_PENDING_NEXT_CANDLE"
            )

    valid_structure_swings = [
        swing for swing in final_confirmed_swings
        if swing["status"] != "PRE_BROKEN"
    ]

    final_highs = [
        swing for swing in final_confirmed_swings
        if swing["type"] == "HIGH"
    ]

    final_lows = [
        swing for swing in final_confirmed_swings
        if swing["type"] == "LOW"
    ]

    usable_highs = [
        swing for swing in final_highs
        if swing["status"] in (
            "ACTIVE",
            "CONFIRMED_PENDING_NEXT_CANDLE"
        )
    ]

    usable_lows = [
        swing for swing in final_lows
        if swing["status"] in (
            "ACTIVE",
            "CONFIRMED_PENDING_NEXT_CANDLE"
        )
    ]

    final_structure = classify_market_structure(
        valid_structure_swings,
        structure_type
    )

    status_counts = {
        status: sum(
            1 for swing in final_confirmed_swings
            if swing["status"] == status
        )
        for status in VALID_SWING_STATUSES
    }

    return {
        "structure_type": structure_type,
        "trend": final_structure["trend"],
        "event_bias": bias,
        "structure": final_structure,
        "swings": final_confirmed_swings,
        "events": events,
        "latest_event": events[-1] if events else None,
        "active_high": (
            usable_highs[-1] if usable_highs else None
        ),
        "active_low": (
            usable_lows[-1] if usable_lows else None
        ),
        "high_count": len(final_highs),
        "low_count": len(final_lows),
        "swing_status_counts": status_counts,
    }


def analyze_daily_structure(
    candles: List[Dict[str, Any]],
    timeframe: str = "1D"
) -> Dict[str, Any]:

    validate_candles(candles)

    candle_duration = infer_candle_duration(
        candles,
        timeframe
    )

    internal_swings = detect_swings(
        candles,
        left_bars=2,
        right_bars=2
    )

    external_swings = detect_swings(
        candles,
        left_bars=5,
        right_bars=5
    )

    internal_result = analyze_structure_events(
        candles,
        internal_swings,
        "INTERNAL",
        candle_duration
    )

    external_result = analyze_structure_events(
        candles,
        external_swings,
        "EXTERNAL",
        candle_duration
    )

    internal_trend = internal_result["event_bias"]
    external_trend = external_result["event_bias"]

    return {
        "timeframe": timeframe,
        "last_candle_timestamp": int(
            float(candles[-1]["timestamp"])
        ),
        "candle_duration_seconds": candle_duration,
        "internal_structure": internal_result,
        "external_structure": external_result,
        "context_direction": {
            "internal": internal_trend,
            "external": external_trend,
        },
        "allowed_direction": "BOTH",
        "trade_directions_to_evaluate": [
            "LONG",
            "SHORT"
        ],
        "status": "ANALYZED",
    }
