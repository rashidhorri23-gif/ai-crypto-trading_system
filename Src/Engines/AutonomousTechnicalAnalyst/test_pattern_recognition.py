"""Unit tests for deterministic chart pattern recognition."""

import numpy as np
import pandas as pd

from .pattern_recognition import detect_chart_patterns


def test_head_and_shoulders_is_detected_from_confirmed_pivots():
    count = 42
    index = np.arange(count)
    close = np.linspace(110.0, 96.0, count)
    frame = pd.DataFrame({
        "timestamp": 1760000000000 + index * 14400000,
        "open": close + 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(count, 100.0),
        "swing_high": np.nan,
        "swing_low": np.nan,
    })

    pivots = [
        (5, "H", 120.0),
        (10, "L", 100.0),
        (15, "H", 140.0),
        (20, "L", 101.0),
        (25, "H", 121.0),
    ]
    for pivot_index, kind, price in pivots:
        confirmation_index = pivot_index + 3
        frame.loc[confirmation_index, "swing_high" if kind == "H" else "swing_low"] = price

    patterns = detect_chart_patterns(frame)
    names = {pattern["name"] for pattern in patterns}
    assert "Head and Shoulders" in names

    pattern = next(item for item in patterns if item["name"] == "Head and Shoulders")
    assert pattern["status"] == "CONFIRMED"
    assert pattern["direction"] == "BEARISH"
    assert pattern["target"] is not None
    assert pattern["invalidation"] == 140.0


def test_insufficient_data_returns_no_patterns():
    frame = pd.DataFrame({
        "timestamp": [1760000000000 + i * 14400000 for i in range(10)],
        "close": [100.0] * 10,
    })
    assert detect_chart_patterns(frame) == []


if __name__ == "__main__":
    test_head_and_shoulders_is_detected_from_confirmed_pivots()
    test_insufficient_data_returns_no_patterns()
    print("CHART PATTERN RECOGNITION TEST PASSED")
