
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


def validate_candles(candles: List[Dict[str, Any]]) -> None:
    if not candles or len(candles) < 20:
        raise ValueError("At least 20 candles are required.")

    previous_timestamp = None

    for index, candle in enumerate(candles):
        for field in REQUIRED_FIELDS:
            if field not in candle:
                raise ValueError(f"Missing field: {field}")

        timestamp = int(float(candle["timestamp"]))

        values = {
            field: float(candle[field])
            for field in REQUIRED_FIELDS[1:]
        }

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

        if high > max(previous_highs + next_highs):
            swings.append({
                "type": "HIGH",
                "index": i,
                "price": high,
                "timestamp": int(
                    float(candles[i]["timestamp"])
                ),
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": int(
                    float(
                        candles[confirmation_index]["timestamp"]
                    )
                ),
            })

        if low < min(previous_lows + next_lows):
            swings.append({
                "type": "LOW",
                "index": i,
                "price": low,
                "timestamp": int(
                    float(candles[i]["timestamp"])
                ),
                "confirmation_index": confirmation_index,
                "confirmation_timestamp": int(
                    float(
                        candles[confirmation_index]["timestamp"]
                    )
                ),
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

    result = {
        "structure_type": structure_type,
        "trend": "UNKNOWN",
        "reason": "Insufficient confirmed swing points.",
        "high_count": len(highs),
        "low_count": len(lows),
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
    }


def analyze_structure_events(
    candles: List[Dict[str, Any]],
    swings: List[Dict[str, Any]],
    structure_type: str,
    candle_duration: int
) -> Dict[str, Any]:

    swings_by_confirmation = {}

    for swing in swings:
        confirmation_index = swing["confirmation_index"]

        swings_by_confirmation.setdefault(
            confirmation_index, []
        ).append(swing)

    active_highs = []
    active_lows = []

    broken_high_indices = set()
    broken_low_indices = set()

    events = []
    bias = "UNKNOWN"

    for i, candle in enumerate(candles):

        # A swing confirmed at candle i becomes usable
        # only from candle i + 1.
        newly_confirmed = swings_by_confirmation.get(
            i - 1, []
        )

        for swing in newly_confirmed:
            if swing["type"] == "HIGH":
                active_highs.append(swing)
            else:
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

        previous_close = float(
            candles[i - 1]["close"]
        )

        current_close = float(candle["close"])

        current_timestamp = int(
            float(candle["timestamp"])
        )

        latest_high = (
            active_highs[-1]
            if active_highs else None
        )

        latest_low = (
            active_lows[-1]
            if active_lows else None
        )

        bullish_break = (
            latest_high is not None
            and latest_high["index"] not in broken_high_indices
            and previous_close <= latest_high["price"]
            and current_close > latest_high["price"]
        )

        bearish_break = (
            latest_low is not None
            and latest_low["index"] not in broken_low_indices
            and previous_close >= latest_low["price"]
            and current_close < latest_low["price"]
        )

        if bullish_break and bearish_break:

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
                "bullish_level": latest_high["price"],
                "bearish_level": latest_low["price"],
                "confirmation": "CANDLE_CLOSE",
                "reason": (
                    "Both structural boundaries were crossed."
                ),
            })

            broken_high_indices.add(
                latest_high["index"]
            )

            broken_low_indices.add(
                latest_low["index"]
            )

            continue

        if bullish_break:

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
                "level": latest_high["price"],
                "level_timestamp": latest_high["timestamp"],
                "level_confirmation_timestamp":
                    latest_high["confirmation_timestamp"],
                "level_available_timestamp":
                    latest_high["confirmation_timestamp"]
                    + candle_duration,
                "previous_close": previous_close,
                "close": current_close,
                "confirmation": "CANDLE_CLOSE",
            })

            broken_high_indices.add(
                latest_high["index"]
            )

            bias = "BULLISH"

        elif bearish_break:

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
                "level": latest_low["price"],
                "level_timestamp": latest_low["timestamp"],
                "level_confirmation_timestamp":
                    latest_low["confirmation_timestamp"],
                "level_available_timestamp":
                    latest_low["confirmation_timestamp"]
                    + candle_duration,
                "previous_close": previous_close,
                "close": current_close,
                "confirmation": "CANDLE_CLOSE",
            })

            broken_low_indices.add(
                latest_low["index"]
            )

            bias = "BEARISH"

    # Include every swing confirmed by the final candle close.
    final_confirmed_swings = [
        swing for swing in swings
        if swing["confirmation_index"] <= len(candles) - 1
    ]

    final_highs = [
        swing for swing in final_confirmed_swings
        if swing["type"] == "HIGH"
    ]

    final_lows = [
        swing for swing in final_confirmed_swings
        if swing["type"] == "LOW"
    ]

    final_structure = classify_market_structure(
        final_confirmed_swings,
        structure_type
    )

    return {
        "structure_type": structure_type,
        "trend": final_structure["trend"],
        "event_bias": bias,
        "structure": final_structure,
        "swings": final_confirmed_swings,
        "events": events,
        "latest_event": events[-1] if events else None,
        "active_high": final_highs[-1] if final_highs else None,
        "active_low": final_lows[-1] if final_lows else None,
        "high_count": len(final_highs),
        "low_count": len(final_lows),
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
