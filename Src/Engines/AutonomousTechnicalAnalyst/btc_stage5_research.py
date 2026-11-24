"""Offline BTC scenario research. This module never places orders or changes live rules."""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

TIMEFRAME_SECONDS = {"5m": 300, "15m": 900, "4H": 14400}
HORIZONS = {"1H": 3600, "4H": 14400, "12H": 43200, "24H": 86400}
RSI_THRESHOLDS = (45.0, 50.0, 55.0)


def load_candles(path, interval):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candle = {
                "timestamp": int(row["timestamp"]) // 1000 if int(row["timestamp"]) > 10**12 else int(row["timestamp"]),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row.get("volume") or 0),
            }
            values = [candle[k] for k in ("open", "high", "low", "close", "volume")]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Non-finite values in {path}")
            if min(values[:4]) <= 0 or candle["high"] < max(candle["open"], candle["close"], candle["low"]) or candle["low"] > min(candle["open"], candle["close"]):
                raise ValueError(f"Invalid OHLC geometry in {path}")
            rows.append(candle)
    rows.sort(key=lambda item: item["timestamp"])
    if not rows:
        raise ValueError(f"No candles found in {path}")
    for previous, current in zip(rows, rows[1:]):
        if current["timestamp"] <= previous["timestamp"]:
            raise ValueError(f"Duplicate or unordered timestamps in {path}")
    return rows


def ema_series(values, period):
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def rsi_series(values, period=14):
    result = [None] * len(values)
    if len(values) <= period:
        return result
    gains, losses = [], []
    for previous, current in zip(values[:-1], values[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def value(gain, loss):
        if loss == 0:
            return 100.0 if gain > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + gain / loss)

    result[period] = value(avg_gain, avg_loss)
    for index in range(period + 1, len(values)):
        gain, loss = gains[index - 1], losses[index - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        result[index] = value(avg_gain, avg_loss)
    return result


def momentum_series(candles):
    closes = [item["close"] for item in candles]
    ema50 = ema_series(closes, 50)
    ema200 = ema_series(closes, 200)
    rsi = rsi_series(closes, 14)
    fast, slow = ema_series(closes, 12), ema_series(closes, 26)
    macd = [fast[i] - slow[i] for i in range(len(closes))]
    signal = ema_series(macd, 9)
    histogram = [macd[i] - signal[i] for i in range(len(closes))]
    snapshots = []
    for i, close in enumerate(closes):
        known = i >= 204 and rsi[i] is not None
        trend = "BULLISH" if ema50[i] > ema200[i] and close > ema50[i] else (
            "BEARISH" if ema50[i] < ema200[i] and close < ema50[i] else "NEUTRAL"
        )
        snapshots.append({
            "known": known, "trend": trend, "rsi": rsi[i],
            "macd_histogram": histogram[i], "close": close,
        })
    return snapshots


def is_aligned(snapshot, direction, threshold):
    if not snapshot or not snapshot["known"]:
        return False
    if direction == "LONG":
        return snapshot["trend"] == "BULLISH" and snapshot["rsi"] > threshold and snapshot["macd_histogram"] > 0
    return snapshot["trend"] == "BEARISH" and snapshot["rsi"] < (100.0 - threshold) and snapshot["macd_histogram"] < 0


def evaluate_variant(five, fifteen, four_hour, threshold):
    five_momentum = momentum_series(five)
    fifteen_momentum = momentum_series(fifteen)
    fifteen_close_times = [c["timestamp"] + TIMEFRAME_SECONDS["15m"] for c in fifteen]
    four_close_times = [c["timestamp"] + TIMEFRAME_SECONDS["4H"] for c in four_hour]
    signals = []
    seen_session_directions = set()
    j15 = -1
    j4h = -1

    for i, candle in enumerate(five):
        close_time = candle["timestamp"] + TIMEFRAME_SECONDS["5m"]
        while j15 + 1 < len(fifteen_close_times) and fifteen_close_times[j15 + 1] <= close_time:
            j15 += 1
        while j4h + 1 < len(four_close_times) and four_close_times[j4h + 1] <= close_time:
            j4h += 1
        if j4h < 0 or j15 < 0:
            continue
        # The daily BTC report is scheduled for 08:00 Baku (04:00 UTC).
        session_start = close_time - ((close_time - 4 * 3600) % 86400)
        baseline_index = -1
        while baseline_index + 1 < len(four_close_times) and four_close_times[baseline_index + 1] <= session_start:
            baseline_index += 1
        if baseline_index < 19 or four_close_times[j4h] <= session_start:
            continue
        previous_20 = four_hour[baseline_index - 19:baseline_index + 1]
        long_level = max(item["high"] for item in previous_20)
        short_level = min(item["low"] for item in previous_20)
        current_4h = four_hour[j4h]
        long_breakout = current_4h["close"] > long_level
        short_breakout = current_4h["close"] < short_level
        if not long_breakout and not short_breakout:
            continue
        for direction, breakout in (("LONG", long_breakout), ("SHORT", short_breakout)):
            key = (session_start, direction)
            if not breakout or key in seen_session_directions:
                continue
            if is_aligned(fifteen_momentum[j15], direction, threshold) and is_aligned(five_momentum[i], direction, threshold):
                signals.append({
                    "direction": direction, "entry_timestamp": close_time,
                    "entry_price": candle["close"],
                    "breakout_candle_close_timestamp": four_close_times[j4h],
                    "frozen_breakout_level": long_level if direction == "LONG" else short_level,
                    "daily_session_start_timestamp": session_start,
                    "rsi_threshold": threshold,
                })
                seen_session_directions.add(key)

    for signal in signals:
        signal["horizons"] = {}
        entry_time = signal["entry_timestamp"]
        entry_price = signal["entry_price"]
        direction = signal["direction"]
        for label, seconds in HORIZONS.items():
            target_time = entry_time + seconds
            future = [c for c in five if c["timestamp"] >= entry_time and c["timestamp"] + 300 <= target_time]
            resolved = next((c for c in five if c["timestamp"] + 300 >= target_time), None)
            if not future or resolved is None:
                signal["horizons"][label] = {"resolved": False}
                continue
            highs = [c["high"] for c in future]
            lows = [c["low"] for c in future]
            if direction == "LONG":
                mfe = (max(highs) - entry_price) / entry_price * 100.0
                mae = (entry_price - min(lows)) / entry_price * 100.0
                forward = (resolved["close"] - entry_price) / entry_price * 100.0
            else:
                mfe = (entry_price - min(lows)) / entry_price * 100.0
                mae = (max(highs) - entry_price) / entry_price * 100.0
                forward = (entry_price - resolved["close"]) / entry_price * 100.0
            signal["horizons"][label] = {
                "resolved": True, "forward_return_pct": forward,
                "mfe_pct": max(0.0, mfe), "mae_pct": max(0.0, mae),
            }
    return signals


def summarize(signals):
    result = {"signal_count": len(signals), "directions": {}, "horizons": {}}
    for direction in ("LONG", "SHORT"):
        subset = [item for item in signals if item["direction"] == direction]
        result["directions"][direction] = {"signal_count": len(subset)}
    for horizon in HORIZONS:
        resolved = [
            item["horizons"][horizon] for item in signals
            if item["horizons"][horizon]["resolved"]
        ]
        returns = [item["forward_return_pct"] for item in resolved]
        result["horizons"][horizon] = {
            "resolved_count": len(resolved),
            "positive_return_count": sum(value > 0 for value in returns),
            "positive_return_rate_pct": round(sum(value > 0 for value in returns) / len(returns) * 100.0, 4) if returns else None,
            "average_forward_return_pct": sum(returns) / len(returns) if returns else None,
            "average_mfe_pct": sum(item["mfe_pct"] for item in resolved) / len(resolved) if resolved else None,
            "average_mae_pct": sum(item["mae_pct"] for item in resolved) / len(resolved) if resolved else None,
        }
    return result


def run_research(data_dir, output_path):
    data_dir = Path(data_dir)
    five = load_candles(data_dir / "BTC_USDT_SPOT_5m.csv", "5m")
    fifteen = load_candles(data_dir / "BTC_USDT_SPOT_15m.csv", "15m")
    four_hour = load_candles(data_dir / "BTC_USDT_SPOT_4H.csv", "4H")
    variants = {}
    all_signals = {}
    for threshold in RSI_THRESHOLDS:
        key = f"RSI_{int(threshold)}"
        signals = evaluate_variant(five, fifteen, four_hour, threshold)
        all_signals[key] = signals
        variants[key] = summarize(signals)
    baseline = all_signals["RSI_50"]
    report = {
        "schema_version": "1.0",
        "research_type": "OFFLINE_HISTORICAL_SCENARIO_FILTER_SENSITIVITY",
        "symbol": "BTC-USDT",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "source": "Repository archived OKX SPOT candles",
            "five_minute_candles": len(five), "fifteen_minute_candles": len(fifteen),
            "four_hour_candles": len(four_hour),
            "five_minute_start": datetime.fromtimestamp(five[0]["timestamp"], timezone.utc).isoformat(),
            "five_minute_end": datetime.fromtimestamp(five[-1]["timestamp"] + 300, timezone.utc).isoformat(),
            "fifteen_minute_start": datetime.fromtimestamp(fifteen[0]["timestamp"], timezone.utc).isoformat(),
            "four_hour_start": datetime.fromtimestamp(four_hour[0]["timestamp"], timezone.utc).isoformat(),
        },
        "rules": {
            "breakout": "Daily frozen breakout levels use the highest/lowest of the 20 completed 4H candles available at 08:00 Baku (04:00 UTC); a later confirmed 4H close must break that frozen level.",
            "momentum": "Both latest confirmed 15M and 5M snapshots require EMA50/EMA200 trend alignment, RSI threshold, and MACD histogram sign.",
            "baseline": "RSI > 50 for LONG and RSI < 50 for SHORT on both timeframes; other live filters unchanged.",
            "variants": "RSI thresholds 45, 50, and 55; short threshold is symmetric (100 - threshold).",
            "forward_windows": "1H, 4H, 12H, and 24H direction-adjusted close returns with MFE/MAE from confirmed 5M candles.",
            "limitations": "Research sample is limited by archived 5M history. Historical daily sessions are anchored at 04:00 UTC to match the 08:00 Baku schedule. Results are descriptive, may contain correlated signals, and are not proof of profitability.",
            "live_system_changed": False,
            "orders_submitted": False,
        },
        "variants": variants,
        "baseline_vs_variants": {
            key: {
                "signal_count_delta": variants[key]["signal_count"] - variants["RSI_50"]["signal_count"],
                "four_hour_positive_return_rate_pct": variants[key]["horizons"]["4H"]["positive_return_rate_pct"],
                "average_4h_forward_return_pct": variants[key]["horizons"]["4H"]["average_forward_return_pct"],
            }
            for key in variants
        },
        "signals": all_signals,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run offline BTC scenario filter research.")
    parser.add_argument("--data-dir", default="Data/Archive")
    parser.add_argument("--output", default="Data/Research/btc_stage5_research_report.json")
    args = parser.parse_args()
    report = run_research(args.data_dir, args.output)
    print(json.dumps({
        "output": args.output,
        "data": report["data"],
        "variants": report["variants"],
        "live_system_changed": False,
        "orders_submitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
