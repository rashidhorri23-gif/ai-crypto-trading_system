"""Candle-based heuristic chart pattern recognition.

Uses confirmed swing points and closed OHLCV candles only. Pattern confidence
is a structural fit score, not a statistically calibrated win probability.
"""
import math


def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _near(a, b, tolerance=0.025):
    return abs(a - b) / max(abs(a), abs(b), 1e-9) <= tolerance


def _make(name, direction, status, confidence, points, lines,
          target, invalidation, trigger):
    return {
        "name": name, "direction": direction, "status": status,
        "confidence": confidence, "points": points, "lines": lines,
        "target": _num(target), "invalidation": _num(invalidation),
        "trigger": trigger,
    }


def _swings(frame, window):
    highs, lows = [], []
    rows = frame.reset_index(drop=True)
    for index, row in rows.iterrows():
        pivot_index = index - window
        if pivot_index < 0:
            continue
        pivot_time = int(_num(rows.iloc[pivot_index]["timestamp"]) // 1000)
        high, low = _num(row.get("swing_high")), _num(row.get("swing_low"))
        if high is not None:
            highs.append({"time": pivot_time, "price": high, "kind": "H"})
        if low is not None:
            lows.append({"time": pivot_time, "price": low, "kind": "L"})
    return highs, lows


def _alternate(highs, lows):
    result = []
    for point in sorted(highs + lows, key=lambda item: item["time"]):
        if result and result[-1]["kind"] == point["kind"]:
            better = point["price"] > result[-1]["price"] if point["kind"] == "H" else point["price"] < result[-1]["price"]
            if better:
                result[-1] = point
        else:
            result.append(point)
    return result


def _head_shoulders(swings, close, inverse=False):
    if len(swings) < 5:
        return None
    seq = swings[-5:]
    expected = ["L", "H", "L", "H", "L"] if inverse else ["H", "L", "H", "L", "H"]
    if [point["kind"] for point in seq] != expected:
        return None
    left, neck_a, head, neck_b, right = seq
    shoulders = _near(left["price"], right["price"], 0.06)
    head_valid = (
        head["price"] < min(left["price"], right["price"]) * 0.975
        if inverse else head["price"] > max(left["price"], right["price"]) * 1.025
    )
    if not shoulders or not head_valid:
        return None
    slope = (neck_b["price"] - neck_a["price"]) / max(1, neck_b["time"] - neck_a["time"])
    neckline = neck_b["price"] + slope * max(0, close["time"] - neck_b["time"])
    confirmed = close["price"] > neckline if inverse else close["price"] < neckline
    height = abs(head["price"] - neckline)
    labels = ["LS", "NL", "H", "NR", "RS"]
    points = [{"time": p["time"], "price": p["price"], "label": label} for p, label in zip(seq, labels)]
    line = {"name": "Neckline", "points": [
        {"time": neck_a["time"], "value": neck_a["price"]},
        {"time": neck_b["time"], "value": neck_b["price"]},
    ]}
    if inverse:
        return _make("Inverse Head and Shoulders", "BULLISH",
            "CONFIRMED" if confirmed else "DEVELOPING", 82 if confirmed else 68,
            points, [line], neckline + height, head["price"],
            "بسته‌شدن کندل بالای خط گردن")
    return _make("Head and Shoulders", "BEARISH",
        "CONFIRMED" if confirmed else "DEVELOPING", 82 if confirmed else 68,
        points, [line], neckline - height, head["price"],
        "بسته‌شدن کندل زیر خط گردن")


def _double_pattern(highs, lows, close):
    if len(highs) >= 2:
        a, b = highs[-2:]
        valley = [p for p in lows if a["time"] < p["time"] < b["time"]]
        if b["time"] > a["time"] and _near(a["price"], b["price"], 0.018) and valley:
            neck = min(valley, key=lambda p: p["price"])
            confirmed = close["price"] < neck["price"]
            height = min(a["price"], b["price"]) - neck["price"]
            return _make("Double Top", "BEARISH", "CONFIRMED" if confirmed else "DEVELOPING",
                78 if confirmed else 62,
                [{"time": a["time"], "price": a["price"], "label": "T1"},
                 {"time": neck["time"], "price": neck["price"], "label": "Neck"},
                 {"time": b["time"], "price": b["price"], "label": "T2"}],
                [{"name": "Neckline", "points": [
                    {"time": a["time"], "value": neck["price"]},
                    {"time": b["time"], "value": neck["price"]}]}],
                neck["price"] - height, max(a["price"], b["price"]),
                "شکست کف بین دو سقف")
    if len(lows) >= 2:
        a, b = lows[-2:]
        peak = [p for p in highs if a["time"] < p["time"] < b["time"]]
        if b["time"] > a["time"] and _near(a["price"], b["price"], 0.018) and peak:
            neck = max(peak, key=lambda p: p["price"])
            confirmed = close["price"] > neck["price"]
            height = neck["price"] - max(a["price"], b["price"])
            return _make("Double Bottom", "BULLISH", "CONFIRMED" if confirmed else "DEVELOPING",
                78 if confirmed else 62,
                [{"time": a["time"], "price": a["price"], "label": "B1"},
                 {"time": neck["time"], "price": neck["price"], "label": "Neck"},
                 {"time": b["time"], "price": b["price"], "label": "B2"}],
                [{"name": "Neckline", "points": [
                    {"time": a["time"], "value": neck["price"]},
                    {"time": b["time"], "value": neck["price"]}]}],
                neck["price"] + height, min(a["price"], b["price"]),
                "شکست سقف بین دو کف")
    return None


def _triangle(highs, lows, close):
    if len(highs) < 3 or len(lows) < 3:
        return None
    upper, lower = highs[-3:], lows[-3:]
    up_slope = (upper[-1]["price"] - upper[0]["price"]) / max(1, upper[-1]["time"] - upper[0]["time"])
    low_slope = (lower[-1]["price"] - lower[0]["price"]) / max(1, lower[-1]["time"] - lower[0]["time"])
    if up_slope >= 0 or low_slope <= 0:
        return None
    upper_now = upper[-1]["price"] + up_slope * max(0, close["time"] - upper[-1]["time"])
    lower_now = lower[-1]["price"] + low_slope * max(0, close["time"] - lower[-1]["time"])
    if upper_now <= lower_now or upper_now - lower_now > close["price"] * 0.12:
        return None
    direction = "BULLISH" if close["price"] > upper_now else "BEARISH" if close["price"] < lower_now else "NEUTRAL"
    status = "CONFIRMED" if direction != "NEUTRAL" else "DEVELOPING"
    height = upper[0]["price"] - lower[0]["price"]
    return _make("Symmetrical Triangle", direction, status, 72 if status == "CONFIRMED" else 58,
        [{"time": p["time"], "price": p["price"], "label": p["kind"]} for p in upper + lower],
        [{"name": "Upper boundary", "points": [
            {"time": upper[0]["time"], "value": upper[0]["price"]},
            {"time": upper[-1]["time"], "value": upper[-1]["price"]}]},
         {"name": "Lower boundary", "points": [
            {"time": lower[0]["time"], "value": lower[0]["price"]},
            {"time": lower[-1]["time"], "value": lower[-1]["price"]}]}],
        close["price"] + (height if direction == "BULLISH" else -height),
        lower_now if direction == "BULLISH" else upper_now,
        "بسته‌شدن کندل خارج از یکی از دو ضلع مثلث")


def _trend_patterns(highs, lows, close):
    if len(highs) < 3 or len(lows) < 3:
        return []
    upper, lower = highs[-3:], lows[-3:]
    upper_slope = (upper[-1]["price"] - upper[0]["price"]) / max(1, upper[-1]["time"] - upper[0]["time"])
    lower_slope = (lower[-1]["price"] - lower[0]["price"]) / max(1, lower[-1]["time"] - lower[0]["time"])
    upper_now = upper[-1]["price"] + upper_slope * max(0, close["time"] - upper[-1]["time"])
    lower_now = lower[-1]["price"] + lower_slope * max(0, close["time"] - lower[-1]["time"])
    high_flat = _near(upper[0]["price"], upper[-1]["price"], 0.018)
    low_flat = _near(lower[0]["price"], lower[-1]["price"], 0.018)
    results = []
    if high_flat and lower_slope > 0:
        direction = "BULLISH" if close["price"] > upper_now else "NEUTRAL"
        results.append(_make("Ascending Triangle", direction,
            "CONFIRMED" if direction == "BULLISH" else "DEVELOPING",
            76 if direction == "BULLISH" else 60,
            [{"time": p["time"], "price": p["price"], "label": "H"} for p in upper] +
            [{"time": p["time"], "price": p["price"], "label": "L"} for p in lower],
            [{"name": "Resistance", "points": [
                {"time": upper[0]["time"], "value": upper[0]["price"]},
                {"time": upper[-1]["time"], "value": upper[-1]["price"]}]},
             {"name": "Rising support", "points": [
                {"time": lower[0]["time"], "value": lower[0]["price"]},
                {"time": lower[-1]["time"], "value": lower[-1]["price"]}]}],
            upper_now + (upper[0]["price"] - lower[0]["price"]),
            lower_now, "بسته‌شدن کندل بالای مقاومت افقی"))
    if low_flat and upper_slope < 0:
        direction = "BEARISH" if close["price"] < lower_now else "NEUTRAL"
        results.append(_make("Descending Triangle", direction,
            "CONFIRMED" if direction == "BEARISH" else "DEVELOPING",
            76 if direction == "BEARISH" else 60,
            [{"time": p["time"], "price": p["price"], "label": "H"} for p in upper] +
            [{"time": p["time"], "price": p["price"], "label": "L"} for p in lower],
            [{"name": "Falling resistance", "points": [
                {"time": upper[0]["time"], "value": upper[0]["price"]},
                {"time": upper[-1]["time"], "value": upper[-1]["price"]}]},
             {"name": "Support", "points": [
                {"time": lower[0]["time"], "value": lower[0]["price"]},
                {"time": lower[-1]["time"], "value": lower[-1]["price"]}]}],
            lower_now - (upper[0]["price"] - lower[0]["price"]),
            upper_now, "بسته‌شدن کندل زیر حمایت افقی"))
    if upper_slope > 0 and lower_slope > upper_slope:
        direction = "BEARISH" if close["price"] < lower_now else "NEUTRAL"
        results.append(_make("Rising Wedge", direction,
            "CONFIRMED" if direction == "BEARISH" else "DEVELOPING",
            73 if direction == "BEARISH" else 57,
            [{"time": p["time"], "price": p["price"], "label": "H"} for p in upper] +
            [{"time": p["time"], "price": p["price"], "label": "L"} for p in lower],
            [{"name": "Upper wedge", "points": [
                {"time": upper[0]["time"], "value": upper[0]["price"]},
                {"time": upper[-1]["time"], "value": upper[-1]["price"]}]},
             {"name": "Lower wedge", "points": [
                {"time": lower[0]["time"], "value": lower[0]["price"]},
                {"time": lower[-1]["time"], "value": lower[-1]["price"]}]}],
            lower_now - (upper[-1]["price"] - lower[-1]["price"]),
            upper_now, "شکست ضلع پایینی کنج صعودی"))
    if upper_slope < 0 and lower_slope < upper_slope:
        direction = "BULLISH" if close["price"] > upper_now else "NEUTRAL"
        results.append(_make("Falling Wedge", direction,
            "CONFIRMED" if direction == "BULLISH" else "DEVELOPING",
            73 if direction == "BULLISH" else 57,
            [{"time": p["time"], "price": p["price"], "label": "H"} for p in upper] +
            [{"time": p["time"], "price": p["price"], "label": "L"} for p in lower],
            [{"name": "Upper wedge", "points": [
                {"time": upper[0]["time"], "value": upper[0]["price"]},
                {"time": upper[-1]["time"], "value": upper[-1]["price"]}]},
             {"name": "Lower wedge", "points": [
                {"time": lower[0]["time"], "value": lower[0]["price"]},
                {"time": lower[-1]["time"], "value": lower[-1]["price"]}]}],
            upper_now + (upper[-1]["price"] - lower[-1]["price"]),
            lower_now, "شکست ضلع بالایی کنج نزولی"))
    return results


def _flag(frame):
    if len(frame) < 24:
        return None
    recent = frame.tail(24).reset_index(drop=True)
    impulse_start = _num(recent.iloc[0]["close"])
    impulse_end = _num(recent.iloc[9]["close"])
    if not impulse_start or not impulse_end:
        return None
    impulse = (impulse_end - impulse_start) / impulse_start
    consolidation = recent.iloc[10:]
    range_candles = consolidation.iloc[:-1]
    highs = [_num(v) for v in range_candles["high"]]
    lows = [_num(v) for v in range_candles["low"]]
    if abs(impulse) < 0.035 or any(v is None for v in highs + lows):
        return None
    ceiling, floor = max(highs), min(lows)
    if (ceiling - floor) / impulse_end > abs(impulse) * 0.9:
        return None
    direction = "BULLISH" if impulse > 0 else "BEARISH"
    close = _num(recent.iloc[-1]["close"])
    confirmed = close > ceiling if direction == "BULLISH" else close < floor
    times = [int(_num(v) // 1000) for v in range_candles["timestamp"]]
    return _make("Bull Flag" if direction == "BULLISH" else "Bear Flag",
        direction, "CONFIRMED" if confirmed else "DEVELOPING", 70 if confirmed else 55,
        [{"time": times[0], "price": ceiling, "label": "Flag"},
         {"time": times[-1], "price": floor, "label": "Flag"}],
        [{"name": "Flag upper", "points": [
            {"time": times[0], "value": ceiling}, {"time": times[-1], "value": ceiling}]},
         {"name": "Flag lower", "points": [
            {"time": times[0], "value": floor}, {"time": times[-1], "value": floor}]}],
        impulse_end + (impulse_end - impulse_start),
        floor if direction == "BULLISH" else ceiling,
        "شکست محدوده پرچم در جهت حرکت قبلی همراه با حجم")



def _pennant(frame):
    """Detect a compact converging consolidation after a strong impulse leg."""
    if len(frame) < 28:
        return None
    recent = frame.tail(28).reset_index(drop=True)
    impulse_start = _num(recent.iloc[0].get("close"))
    impulse_end = _num(recent.iloc[9].get("close"))
    if not impulse_start or not impulse_end:
        return None
    impulse = (impulse_end - impulse_start) / impulse_start
    if abs(impulse) < 0.035:
        return None

    consolidation = recent.iloc[10:]
    highs = [_num(value) for value in consolidation["high"]]
    lows = [_num(value) for value in consolidation["low"]]
    if any(value is None for value in highs + lows):
        return None

    count = len(consolidation)
    x = list(range(count))
    x_mean = sum(x) / count
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        return None
    high_mean = sum(highs) / count
    low_mean = sum(lows) / count
    high_slope = sum((x[i] - x_mean) * (highs[i] - high_mean) for i in x) / denominator
    low_slope = sum((x[i] - x_mean) * (lows[i] - low_mean) for i in x) / denominator
    upper_now = highs[-1] + high_slope * (count - 1)
    lower_now = lows[-1] + low_slope * (count - 1)
    initial_width = max(highs[0] - lows[0], 1e-9)
    final_width = max(highs[-1] - lows[-1], 0)
    if high_slope >= 0 or low_slope <= 0:
        return None
    if final_width >= initial_width * 0.8:
        return None
    if upper_now <= lower_now:
        return None

    direction = "BULLISH" if impulse > 0 else "BEARISH"
    last_close = _num(recent.iloc[-1].get("close"))
    if last_close is None:
        return None
    confirmed = last_close > upper_now if direction == "BULLISH" else last_close < lower_now
    timestamps = [_num(value) for value in consolidation["timestamp"]]
    if any(value is None for value in timestamps):
        return None
    start_time = int(timestamps[0] // 1000)
    end_time = int(timestamps[-1] // 1000)
    upper_start, upper_end = highs[0], highs[-1]
    lower_start, lower_end = lows[0], lows[-1]
    target = last_close + (impulse_end - impulse_start)
    invalidation = lower_end if direction == "BULLISH" else upper_end

    return _make(
        "Bullish Pennant" if direction == "BULLISH" else "Bearish Pennant",
        direction,
        "CONFIRMED" if confirmed else "DEVELOPING",
        74 if confirmed else 59,
        [
            {"time": start_time, "price": upper_start, "label": "P-U"},
            {"time": end_time, "price": upper_end, "label": "P-U"},
            {"time": start_time, "price": lower_start, "label": "P-L"},
            {"time": end_time, "price": lower_end, "label": "P-L"},
        ],
        [
            {"name": "Pennant upper", "points": [
                {"time": start_time, "value": upper_start},
                {"time": end_time, "value": upper_end},
            ]},
            {"name": "Pennant lower", "points": [
                {"time": start_time, "value": lower_start},
                {"time": end_time, "value": lower_end},
            ]},
        ],
        target,
        invalidation,
        "بسته‌شدن کندل خارج از محدوده همگرا در جهت میله پرچم",
    )

def detect_chart_patterns(frame, swing_window=3):
    """Detect a small set of interpretable patterns on the supplied timeframe."""
    if frame is None or len(frame) < 30:
        return []
    highs, lows = _swings(frame, swing_window)
    swings = _alternate(highs, lows)
    latest = frame.iloc[-1]
    close = {"time": int(_num(latest["timestamp"]) // 1000), "price": _num(latest["close"])}
    if close["price"] is None:
        return []
    patterns = [
        _head_shoulders(swings, close, False),
        _head_shoulders(swings, close, True),
        _double_pattern(highs, lows, close),
        _triangle(highs, lows, close),
        *_trend_patterns(highs, lows, close),
        _flag(frame),
        _pennant(frame),
    ]
    patterns = [p for p in patterns if p is not None]
    patterns.sort(key=lambda p: (p["status"] != "CONFIRMED", -p["confidence"]))
    return patterns[:8]
