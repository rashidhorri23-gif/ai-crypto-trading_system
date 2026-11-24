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
from io import BytesIO

import requests
from zoneinfo import ZoneInfo

from Src.Engines.AutonomousTechnicalAnalyst import btc_scenario_performance as performance

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

def report_breakout_levels(report):
    """Read the frozen 20-candle levels published with the daily report."""
    try:
        four_hour = report["forecast"]["timeframes"]["4H"]
        long_level = float(four_hour["bullish"]["breakout_level"])
        short_level = float(four_hour["bearish"]["breakout_level"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Daily report is missing frozen 4H breakout levels. "
            "Refusing to calculate independent monitor levels."
        ) from error
    if not all(math.isfinite(value) and value > 0 for value in (long_level, short_level)):
        raise ValueError("Daily report contains invalid 4H breakout levels.")
    if long_level <= short_level:
        raise ValueError("Daily report 4H breakout levels have invalid ordering.")
    return {"LONG": long_level, "SHORT": short_level}


def okx_candles(bar):
    collected = {}
    before = None
    for page_number in range(3):
        params = {"instId": SYMBOL, "bar": bar, "limit": "100"}
        if before is not None:
            params["after"] = str(before)
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{OKX_BASE_URL}/api/v5/market/history-candles?{query}",
            headers={"Accept": "application/json", "User-Agent": "BTC-Dual-Scenario-Monitor/1.0"},
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("code") != "0":
            raise RuntimeError(f"OKX API error for {bar}: {payload.get('msg', 'unknown')}")
        rows = payload.get("data", [])
        if not rows:
            break
        oldest_ms = min(int(row[0]) for row in rows)
        for row in rows:
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
            collected[item["timestamp"]] = item
        if before == oldest_ms:
            break
        before = oldest_ms
        time.sleep(0.1)
    return [collected[key] for key in sorted(collected)]


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


def derive_breakout_levels(candles, generated_at):
    baseline = [
        candle for candle in candles
        if candle["timestamp"] + TIMEFRAMES["4H"][1] <= generated_at
    ]
    if len(baseline) < 21:
        raise ValueError("At least 21 confirmed pre-report 4H candles are required.")
    previous_20 = baseline[-21:-1]
    return {
        "LONG": max(candle["high"] for candle in previous_20),
        "SHORT": min(candle["low"] for candle in previous_20),
    }


def evaluate_scenarios(candles_by_tf, generated_at, breakout_levels=None):
    four_hour = candles_by_tf["4H"]
    fifteen = candles_by_tf["15M"]
    five = candles_by_tf["5M"]
    if len(four_hour) < 21:
        raise ValueError("At least 21 confirmed 4H candles are required.")
    latest_4h = four_hour[-1]
    close_4h_timestamp = latest_4h["timestamp"] + TIMEFRAMES["4H"][1]
    levels = breakout_levels or derive_breakout_levels(four_hour, generated_at)
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
        breakout_level = float(levels[direction])
        breakout_met = (
            four_hour_fresh
            and (latest_4h["close"] > breakout_level if bullish else latest_4h["close"] < breakout_level)
        )
        checks = {
            "4H_BREAKOUT": {
                "met": bool(breakout_met),
                "known": four_hour_fresh,
                "close": latest_4h["close"],
                "level": breakout_level,
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



def build_status_chart(candles, session):
    """Render a spacious candlestick chart from confirmed OKX 5-minute candles."""
    if not isinstance(candles, list) or len(candles) < 2:
        raise ValueError("At least two confirmed 5-minute candles are required for the status chart.")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    selected = candles[-48:]
    opens = [float(item["open"]) for item in selected]
    highs = [float(item["high"]) for item in selected]
    lows = [float(item["low"]) for item in selected]
    closes = [float(item["close"]) for item in selected]

    figure = Figure(figsize=(16, 9), dpi=180)
    axis = figure.add_subplot(1, 1, 1)

    for index, (open_price, high, low, close) in enumerate(zip(opens, highs, lows, closes)):
        rising = close >= open_price
        candle_color = "green" if rising else "red"
        axis.vlines(index, low, high, color=candle_color, linewidth=1.6, zorder=2)
        body_bottom = min(open_price, close)
        body_height = max(abs(close - open_price), max(highs) * 0.000025)
        axis.add_patch(
            Rectangle(
                (index - 0.32, body_bottom),
                0.64,
                body_height,
                facecolor=candle_color,
                edgecolor=candle_color,
                linewidth=0.8,
                alpha=0.85,
                zorder=3,
            )
        )

    axis.scatter(len(closes) - 1, closes[-1], s=55, marker="o",
                 edgecolors="black", linewidths=0.8, zorder=5,
                 label=f"Latest close: {closes[-1]:,.2f} USDT")

    price_min = min(lows)
    price_max = max(highs)
    price_span = max(price_max - price_min, price_max * 0.001)
    padding = price_span * 0.12
    visible_min = price_min - padding
    visible_max = price_max + padding
    axis.set_ylim(visible_min, visible_max)
    axis.set_xlim(-1, len(selected))

    levels = session.get("breakout_levels") or {}
    level_descriptions = []
    for direction, color in (("LONG", "green"), ("SHORT", "red")):
        level = levels.get(direction)
        if level is None:
            level_descriptions.append(f"{direction}: unavailable")
            continue
        level = float(level)
        location = ""
        if visible_min <= level <= visible_max:
            axis.axhline(
                level,
                linestyle="--",
                linewidth=2.0,
                color=color,
                alpha=0.9,
                label=f"{direction} breakout: {level:,.2f}",
            )
        elif level > visible_max:
            location = " (above visible price range)"
        else:
            location = " (below visible price range)"
        level_descriptions.append(f"{direction}: {level:,.2f}{location}")

    tick_positions = sorted(set([0, len(selected) // 4, len(selected) // 2,
                                 (3 * len(selected)) // 4, len(selected) - 1]))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [datetime.fromtimestamp(selected[index]["timestamp"], timezone.utc).strftime("%H:%M")
         for index in tick_positions],
        fontsize=12,
    )
    axis.tick_params(axis="y", labelsize=12)
    axis.set_title("BTC-USDT | 5-Minute Candlestick Chart", fontsize=19, pad=18)
    axis.set_ylabel("Price (USDT)", fontsize=14, labelpad=10)
    axis.set_xlabel("Time (UTC)", fontsize=14, labelpad=10)
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.legend(loc="best", fontsize=11, framealpha=0.9)
    figure.text(
        0.5, 0.025, "   |   ".join(level_descriptions),
        ha="center", va="bottom", fontsize=12,
    )
    figure.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.13)

    output = BytesIO()
    FigureCanvasAgg(figure).print_png(output)
    return output.getvalue()

def send_telegram_photo(image_bytes, caption, event_id, state):
    sent_ids = state.setdefault("notification_ids", [])
    if event_id in sent_ids:
        return True
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram credentials are not configured; photo notification remains pending.")
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption[:1024]},
            files={"photo": ("btc_monitor_status.png", image_bytes, "image/png")},
            timeout=45,
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            print(f"Telegram rejected monitor chart: {result}")
            return False
        sent_ids.append(event_id)
        state["notification_ids"] = sent_ids[-500:]
        return True
    except Exception as error:
        print(f"Telegram chart notification failed: {error}")
        return False


def telegram_get_updates(offset):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return []
    params = {"timeout": "0", "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        params["offset"] = str(int(offset))
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/getUpdates?{urllib.parse.urlencode(params)}",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(f"Telegram command polling failed: {error}")
        return []
    if not payload.get("ok"):
        print(f"Telegram command polling returned an API error: {payload.get('description', 'unknown')}")
        return []
    return payload.get("result", [])


def poll_telegram_commands(state):
    """Accept status controls only from the configured Telegram chat."""
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not chat_id:
        return
    updates = telegram_get_updates(state.get("telegram_update_offset"))
    for update in updates:
        update_id = int(update.get("update_id", -1))
        if update_id < 0:
            continue
        state["telegram_update_offset"] = update_id + 1
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        if str(chat.get("id", "")) != str(chat_id):
            continue
        command_text = str(message.get("text", "")).strip().split()
        if not command_text:
            continue
        command = command_text[0].split("@")[0].lower()
        if command in ("/start", "/help"):
            reply = (
                "BTC scenario monitor commands:\n"
                "/status_on - Enable periodic BTC status messages.\n"
                "/status_off - Disable periodic status messages.\n"
                "/status - Show the current setting.\n"
                "LONG/SHORT activation alerts remain enabled either way."
            )
        elif command == "/status_on":
            state["status_updates_enabled"] = True
            reply = "BTC status updates are ON. You will receive one concise update per new 5-minute closed candle (approximately every 15-minute monitor cycle)."
        elif command == "/status_off":
            state["status_updates_enabled"] = False
            reply = "BTC status updates are OFF. LONG/SHORT activation alerts remain enabled."
        elif command == "/status":
            enabled = bool(state.get("status_updates_enabled", False))
            reply = (
                "BTC periodic status: ON."
                if enabled else
                "BTC periodic status: OFF. Use /status_on to enable or /status_off to disable."
            )
        else:
            continue
        event_id = notification_id("status_command", str(update_id))
        send_telegram(reply, event_id, state)


def send_periodic_status(session, state, candles_by_tf=None):
    if session.get("status") != "ACTIVE":
        return
    if not state.get("status_updates_enabled", False):
        return
    checks = session.get("latest_checks")
    if not isinstance(checks, dict):
        return
    observed = max(
        (int(item.get("observed_timestamp") or 0) for item in checks.values()),
        default=0,
    )
    if observed <= 0:
        return
    if state.get("status_update_last_candle") == observed:
        return
    price = next(
        (item.get("observed_price") for item in checks.values() if item.get("observed_price") is not None),
        None,
    )
    lines = [
        "BTC MONITOR STATUS",
        f"Price: {float(price):,.2f} USDT" if price is not None else "Price: unavailable",
        f"Checked: {utc_iso(observed)}",
    ]
    for direction in ("LONG", "SHORT"):
        scenario = session.get("scenarios", {}).get(direction, {})
        evaluation = checks.get(direction, {})
        status = scenario.get("status", "WAITING")
        checks_map = evaluation.get("checks", {})
        confirmations = sum(1 for item in checks_map.values() if item.get("known") and item.get("met"))
        lines.append(f"{direction}: {status} ({confirmations}/3 conditions confirmed)")
    lines.append("Monitoring only. No trade was opened.")
    message = "\n".join(lines)
    event_id = notification_id("periodic_status", session["analysis_id"], str(observed))

    five_minute = (candles_by_tf or {}).get("5M", [])
    sent = False
    if len(five_minute) >= 2:
        try:
            chart = build_status_chart(five_minute, session)
            sent = send_telegram_photo(chart, message, event_id, state)
        except Exception as error:
            print(f"Unable to build BTC monitor chart: {error}")
    if not sent:
        sent = send_telegram(message, event_id, state)
    if sent:
        state["status_update_last_candle"] = observed


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

    if not isinstance(current.get("breakout_levels"), dict):
        current["breakout_levels"] = report_breakout_levels(report)
    else:
        published_levels = report_breakout_levels(report)
        for direction in ("LONG", "SHORT"):
            if not math.isclose(
                float(current["breakout_levels"].get(direction, float("nan"))),
                published_levels[direction],
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError(
                    f"{direction} breakout level mismatch between active session and daily report."
                )
    evaluations = evaluate_scenarios(
        candles_by_tf,
        float(current["generated_at_timestamp"]),
        current["breakout_levels"],
    )
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
    poll_telegram_commands(state)
    now = utc_now().timestamp()
    candles_by_tf = {tf: okx_candles(bar) for tf, (bar, _) in TIMEFRAMES.items()}
    session = process_session(
        report, generated_at, state, now_timestamp=now, candles_by_tf=candles_by_tf
    )
    send_periodic_status(session, state, candles_by_tf)
    performance.record_observation(session, now_timestamp=now)
    performance.update_outcomes(session, candles_by_tf, now_timestamp=now)
    performance.refresh_performance_report(generated_at=now)
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
