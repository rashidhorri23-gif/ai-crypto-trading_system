
from typing import Dict, List, Any


def validate_candles(candles: List[Dict[str, Any]]) -> None:
    if not candles or len(candles) < 20:
        raise ValueError("At least 20 candles are required.")

    previous_timestamp = None

    for candle in candles:
        required = ("timestamp", "open", "high", "low", "close", "volume")

        for field in required:
            if field not in candle:
                raise ValueError(f"Missing field: {field}")

        timestamp = int(candle["timestamp"])
        open_price = float(candle["open"])
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        volume = float(candle["volume"])

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Candles must be uniquely sorted by timestamp.")

        if min(open_price, high, low, close) <= 0:
            raise ValueError("Prices must be positive.")

        if volume < 0:
            raise ValueError("Volume cannot be negative.")

        if high < max(open_price, close, low):
            raise ValueError("Invalid candle high.")

        if low > min(open_price, close, high):
            raise ValueError("Invalid candle low.")

        previous_timestamp = timestamp


def detect_swings(
    candles: List[Dict[str, Any]],
    left_bars: int = 3,
    right_bars: int = 3
) -> List[Dict[str, Any]]:

    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing windows must be positive.")

    swings = []

    for i in range(left_bars, len(candles) - right_bars):
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

        if high > max(previous_highs + next_highs):
            swings.append({
                "type": "HIGH",
                "index": i,
                "price": high,
                "timestamp": int(candles[i]["timestamp"]),
                "confirmation_index": i + right_bars,
                "confirmation_timestamp": int(
                    candles[i + right_bars]["timestamp"]
                )
            })

        if low < min(previous_lows + next_lows):
            swings.append({
                "type": "LOW",
                "index": i,
                "price": low,
                "timestamp": int(candles[i]["timestamp"]),
                "confirmation_index": i + right_bars,
                "confirmation_timestamp": int(
                    candles[i + right_bars]["timestamp"]
                )
            })

    return sorted(swings, key=lambda item: item["index"])


def classify_market_structure(
    swings: List[Dict[str, Any]],
    structure_type: str
) -> Dict[str, Any]:

    highs = [s for s in swings if s["type"] == "HIGH"]
    lows = [s for s in swings if s["type"] == "LOW"]

    if len(highs) < 2 or len(lows) < 2:
        return {
            "structure_type": structure_type,
            "trend": "UNKNOWN",
            "reason": "Insufficient confirmed swing points."
        }

    previous_high, latest_high = highs[-2], highs[-1]
    previous_low, latest_low = lows[-2], lows[-1]

    higher_high = latest_high["price"] > previous_high["price"]
    higher_low = latest_low["price"] > previous_low["price"]

    lower_high = latest_high["price"] < previous_high["price"]
    lower_low = latest_low["price"] < previous_low["price"]

    if higher_high and higher_low:
        trend = "BULLISH"
    elif lower_high and lower_low:
        trend = "BEARISH"
    else:
        trend = "RANGE"

    return {
        "structure_type": structure_type,
        "trend": trend,
        "previous_high": previous_high,
        "latest_high": latest_high,
        "previous_low": previous_low,
        "latest_low": latest_low,
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low
    }


def analyze_daily_structure(
    candles: List[Dict[str, Any]]
) -> Dict[str, Any]:

    validate_candles(candles)

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

    internal_structure = classify_market_structure(
        internal_swings,
        "INTERNAL"
    )

    external_structure = classify_market_structure(
        external_swings,
        "EXTERNAL"
    )

    if external_structure["trend"] == "UNKNOWN":
        allowed_direction = "WAIT"
    elif internal_structure["trend"] == "UNKNOWN":
        allowed_direction = "WAIT"
    elif internal_structure["trend"] != external_structure["trend"]:
        allowed_direction = "WAIT"
    elif external_structure["trend"] == "BULLISH":
        allowed_direction = "LONG"
    elif external_structure["trend"] == "BEARISH":
        allowed_direction = "SHORT"
    else:
        allowed_direction = "WAIT"

    return {
        "timeframe": "1D",
        "last_candle_timestamp": int(candles[-1]["timestamp"]),
        "internal_structure": internal_structure,
        "external_structure": external_structure,
        "allowed_direction": allowed_direction,
        "status": "ANALYZED"
    }
