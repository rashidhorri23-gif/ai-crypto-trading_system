"""Independent BTC dual-scenario activation monitor.

Checks only confirmed OKX candles. It records LONG/SHORT scenario activation
and sends Telegram alerts; it never creates or submits trading orders.
"""
import hashlib
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[3]
REPORT_FILE = ROOT_DIR / "Data" / "Signals" / "current_btc_signal.json"
STATE_FILE = ROOT_DIR / "Data" / "Signals" / "btc_scenario_monitor_state.json"
EVENTS_FILE = ROOT_DIR / "Data" / "Signals" / "btc_scenario_events.json"
OKX_BASE_URL = os.getenv("OKX_BASE_URL", "https://www.okx.com").rstrip("/")
SYMBOL = "BTC-USDT"
TIMEFRAMES = {"4H": ("4H", 14400), "15M": ("15m", 900), "5M": ("5m", 300)}
REQUEST_TIMEOUT = 20
MAX_CANDLES = 300
MAX_SESSION_HISTORY = 90


def utc_now():
    return datetime.now(timezone.utc)


def parse_timestamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Timestamp is missing.")
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def utc_iso(timestamp=None):
    if timestamp is None:
        return utc_now().isoformat()
    return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()


def next_baku_0800_timestamp(reference_timestamp):
    baku = ZoneInfo("Asia/Baku")
    local_time = datetime.fromtimestamp(reference_timestamp, timezone.utc).astimezone(baku)
    next_run = local_time.replace(hour=8, minute=0, second=0, microsecond=0)
    if local_time >= next_run:
        next_run += timedelta(days=1)
    return next_run.astimezone(timezone.utc).timestamp()


def atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def read_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report():
    report = read_json(REPORT_FILE, None)
    if not isinstance(report, dict):
        raise FileNotFoundError(f"Current BTC report not found: {REPORT_FILE}")
    if report.get("schema_version") != "2.0":
        raise ValueError("Unsupported BTC report schema.")
    if report.get("symbol") not in ("BTC-USDT", "BTC_USDT"):
        raise ValueError("Scenario monitor accepts BTC-USDT only.")
    if not isinstance(report.get("analysis_id"), str) or not report["analysis_id"].strip():
        raise ValueError("Report analysis_id is missing.")
    generated_at = parse_timestamp(report.get("generated_at"))
    now = utc_now().timestamp()
    if generated_at > now + 60 or generated_at < now - 86400:
        raise ValueError("BTC report timestamp is invalid or older than 24 hours.")
    forecast = report.get("forecast")
    if not isinstance(forecast, dict) or not isinstance(forecast.get("timeframes"), dict):
        raise ValueError("BTC report has no multi-timeframe forecast.")
    return report, generated_at


def load_state():
    state = read_json(STATE_FILE, {})
    if not isinstance(state, dict):
        raise ValueError("Scenario monitor state must be an object.")
    state.setdefault("version", 1)
    state.setdefault("current_session", None)
    state.setdefault("history", [])
    state.setdefault("notification_ids", [])
    return state


def okx_candles(bar):
    query = urllib.parse.urlencode({"instId": SYMBOL, "bar": bar, "limit": str(MAX_CANDLES)})
    request = urllib.request.Request(
        f"{OKX_BASE_URL}/api/v5/market/history-candles?{query}",
        headers={"Accept": "application/json", "User-Agent": "BTC-Dual-Scenario-Monitor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX API error for {bar}: {payload.get('msg', 'unknown')}")
    candles = {}
    for row in payload.get("data", []):
        if not isinstance(row, list) or len(row) < 9 or str(row[8]) != "1":
            continue
        item = {
            "timestamp": int(row[0]) // 1000,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        values = [item[k] for k in ("open", "high", "low", "close", "volume")]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite candle returned for {bar}.")
        if min(values[:4]) <= 0 or item["volume"] < 0:
            raise ValueError(f"Invalid candle values returned for {bar}.")
        if item["high"] < max(item["open"], item["close"], item["low"]) or item["low"] > min(item["open"], item["close"]):
            raise ValueError(f"Invalid candle geometry returned for {bar}.")
        candles[item["timestamp"]] = item
    return [candles[key] for key in sorted(candles)]


def ema(values, period):
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for previous, current in zip(values[:-1], values[1:]):
        change = current - previous
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def momentum_snapshot(candles):
    if len(candles) < 205:
        return {"known": False, "reason": "Insufficient closed candles for EMA200."}
    closes = [c["close"] for c in candles]
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    current_rsi = rsi(closes, 14)
    fast = []
    slow = []
    for index in range(len(closes)):
        fast.append(ema(closes[:index + 1], 12))
        slow.append(ema(closes[:index + 1], 26))
    macd_line = [a - b for a, b in zip(fast, slow) if a is not None and b is not None]
    signal_line = ema(macd_line, 9) if len(macd_line) >= 9 else None
    histogram = macd_line[-1] - signal_line if signal_line is not None else None
    if any(value is None for value in (ema50, ema200, current_rsi, histogram)):
        return {"known": False, "reason": "Momentum indicators are unavailable."}
    close = closes[-1]
    trend = "BULLISH" if ema50 > ema200 and close > ema50 else (
        "BEARISH" if ema50 < ema200 and close < ema50 else "NEUTRAL"
    )
    return {
        "known": True,
        "close": close,
        "ema_50": ema50,
        "ema_200": ema200,
        "ema_trend": trend,
        "rsi": current_rsi,
        "macd_histogram": histogram,
        "candle_close_timestamp": candles[-1]["timestamp"] + TIMEFRAMES["5M" if len(candles) == -1 else "5M"][1],
    }


def evaluate_scenarios(candles_by_tf, generated_at):
    four_hour = candles_by_tf["4H"]
    fifteen = candles_by_tf["15M"]
    five = candles_by_tf["5M"]
    if len(four_hour) < 21:
        raise ValueError("At least 21 confirmed 4H candles are required.")
    latest_4h = four_hour[-1]
    close_4h_timestamp = latest_4h["timestamp"] + TIMEFRAMES["4H"][1]
    prior_20 = four_hour[-21:-1]
    breakout_high = max(c["high"] for c in prior_20)
    breakout_low = min(c["low"] for c in prior_20)
    four_hour_fresh = close_4h_timestamp > generated_at
    momentum = {}
    for tf, candles in (("15M", fifteen), ("5M", five)):
        if not candles:
            momentum[tf] = {"known": False, "reason": "No confirmed candles."}
            continue
        interval = TIMEFRAMES[tf][1]
        close_timestamp = candles[-1]["timestamp"] + interval
        snapshot = momentum_snapshot(candles)
        snapshot["candle_close_timestamp"] = close_timestamp
        snapshot["fresh_after_report"] = close_timestamp > generated_at
        momentum[tf] = snapshot

    results = {}
    for direction in ("LONG", "SHORT"):
        bullish = direction == "LONG"
        breakout_met = (
            four_hour_fresh
            and (latest_4h["close"] > breakout_high if bullish else latest_4h["close"] < breakout_low)
        )
        checks = {
            "4H_BREAKOUT": {
                "met": bool(breakout_met),
                "known": four_hour_fresh,
                "close": latest_4h["close"],
                "level": breakout_high if bullish else breakout_low,
                "candle_close_timestamp": close_4h_timestamp,
            }
        }
        for tf in ("15M", "5M"):
            snap = momentum[tf]
            fresh = bool(snap.get("fresh_after_report"))
            known = bool(snap.get("known")) and fresh
            trend_ok = snap.get("ema_trend") == ("BULLISH" if bullish else "BEARISH")
            rsi_value = snap.get("rsi")
            macd_value = snap.get("macd_histogram")
            momentum_ok = (
                known
                and trend_ok
                and (rsi_value > 50 and macd_value > 0 if bullish else rsi_value < 50 and macd_value < 0)
            )
            checks[f"{tf}_MOMENTUM"] = {
                "met": bool(momentum_ok),
                "known": known,
                "close": snap.get("close"),
                "ema_trend": snap.get("ema_trend"),
                "rsi": rsi_value,
                "macd_histogram": macd_value,
                "candle_close_timestamp": snap.get("candle_close_timestamp"),
            }
        results[direction] = {
            "checks": checks,
            "ready": all(item["known"] and item["met"] for item in checks.values()),
            "observed_price": five[-1]["close"] if five else None,
            "observed_timestamp": five[-1]["timestamp"] + TIMEFRAMES["5M"][1] if five else None,
        }
    return results


def notification_id(event, analysis_id, direction=""):
    raw = f"{event}|{analysis_id}|{direction}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def send_telegram(message, event_id, state):
    sent_ids = state.setdefault("notification_ids", [])
    if event_id in sent_ids:
        return True
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials are not configured; notification remains pending.")
        return False
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            return False
        sent_ids.append(event_id)
        state["notification_ids"] = sent_ids[-500:]
        return True
    except Exception as error:
        print(f"Telegram notification failed: {error}")
        return False


def append_history(state, session):
    state.setdefault("history", []).append(session)
    state["history"] = state["history"][-MAX_SESSION_HISTORY:]


def finalize_session(session, state, finalized_at, reason):
    if session.get("status") != "ACTIVE":
        return
    session["finalized_at"] = utc_iso(finalized_at)
    session["finalization_reason"] = reason
    activated = [
        direction for direction in ("LONG", "SHORT")
        if session.get("scenarios", {}).get(direction, {}).get("status") == "ACTIVATED"
    ]
    if activated:
        session["status"] = "COMPLETED_WITH_ACTIVATION"
        session["activated_directions"] = activated
        append_history(state, session)
        return
    session["status"] = "NO_SCENARIO_ACTIVATED"
    append_history(state, session)
    event_id = notification_id("no_scenario", session["analysis_id"])
    message = (
        "BTC DAILY SCENARIO MONITOR — NO SCENARIO ACTIVATED\\n"
        f"Analysis ID: {session['analysis_id']}\\n"
        f"Window ended: {session['finalized_at']}\\n"
        "Neither LONG nor SHORT met all confirmed-candle conditions. No trade was opened."
    )
    if send_telegram(message, event_id, state):
        session["no_scenario_notification_sent"] = True

def start_session(report, generated_at):
    return {
        "analysis_id": report["analysis_id"],
        "generated_at": utc_iso(generated_at),
        "generated_at_timestamp": generated_at,
        "expires_at": next_baku_0800_timestamp(generated_at),
        "status": "ACTIVE",
        "scenarios": {
            "LONG": {"status": "WAITING", "activation": None, "notification_sent": False},
            "SHORT": {"status": "WAITING", "activation": None, "notification_sent": False},
        },
        "last_checked_at": None,
    }


def notify_activation(session, direction, scenario, state):
    activation = scenario["activation"]
    message = (
        f"BTC {direction} SCENARIO ACTIVATED\n"
        f"Price: {activation['price']:.2f} USDT\n"
        f"Confirmed at: {activation['timestamp']}\n"
        f"4H breakout level: {activation['checks']['4H_BREAKOUT']['level']:.2f}\n"
        f"Analysis ID: {session['analysis_id']}\n"
        "Scenario alert only. No trade was opened and no order was submitted."
    )
    event_id = notification_id("scenario_activated", session["analysis_id"], direction)
    if send_telegram(message, event_id, state):
        scenario["notification_sent"] = True


def process_session(report, generated_at, state, now_timestamp=None, candles_by_tf=None):
    now = float(now_timestamp if now_timestamp is not None else utc_now().timestamp())
    current = state.get("current_session")
    analysis_id = report["analysis_id"]

    if current and current.get("analysis_id") != analysis_id:
        if current.get("status") == "ACTIVE":
            finalize_session(current, state, now, "REPLACED_BY_NEW_DAILY_REPORT")
        current = None

    if current is None:
        current = start_session(report, generated_at)
        state["current_session"] = current

    if current.get("status") != "ACTIVE":
        return current

    if now >= float(current["expires_at"]):
        finalize_session(current, state, float(current["expires_at"]), "DAILY_WINDOW_ENDED")
        return current

    if candles_by_tf is None:
        candles_by_tf = {
            tf: okx_candles(bar)
            for tf, (bar, _) in TIMEFRAMES.items()
        }

    evaluations = evaluate_scenarios(candles_by_tf, float(current["generated_at_timestamp"]))
    current["last_checked_at"] = utc_iso(now)
    current["latest_checks"] = evaluations

    for direction in ("LONG", "SHORT"):
        scenario = current["scenarios"][direction]
        if scenario["status"] == "ACTIVATED":
            if not scenario.get("notification_sent"):
                notify_activation(current, direction, scenario, state)
            continue
        evaluation = evaluations[direction]
        if evaluation["ready"]:
            scenario["status"] = "ACTIVATED"
            scenario["activation"] = {
                "timestamp": utc_iso(evaluation["observed_timestamp"]),
                "timestamp_unix": evaluation["observed_timestamp"],
                "price": float(evaluation["observed_price"]),
                "checks": evaluation["checks"],
            }
            notify_activation(current, direction, scenario, state)

    return current


def main():
    report, generated_at = load_report()
    state = load_state()
    now = utc_now().timestamp()
    session = process_session(report, generated_at, state, now_timestamp=now)
    atomic_write_json(STATE_FILE, state)
    events = {
        "symbol": SYMBOL,
        "updated_at": utc_iso(now),
        "current_session": state.get("current_session"),
        "history": state.get("history", []),
    }
    atomic_write_json(EVENTS_FILE, events)
    print(f"Analysis ID: {report['analysis_id']}")
    print(f"Session status: {session['status']}")
    for direction in ("LONG", "SHORT"):
        scenario = session["scenarios"][direction]
        print(f"{direction}: {scenario['status']}")
        if scenario.get("activation"):
            print(f"  Price: {scenario['activation']['price']}")
            print(f"  Time: {scenario['activation']['timestamp']}")
    print("Scenario monitor completed. No exchange orders are submitted.")


if __name__ == "__main__":
    main()
