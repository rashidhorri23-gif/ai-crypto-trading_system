
"""AI-driven BTC paper-trading monitor.

AI is the only source of trading decisions.
This module handles market data, report validation, paper trading,
state persistence, and Telegram notifications.

No real exchange orders are submitted.
"""

import copy
import hashlib
import json
import math
import os
import time
import urllib.parse
import urllib.request

from datetime import datetime, timezone
from pathlib import Path

from Src.Engines.AutonomousTechnicalAnalyst import paper_trading


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

# AI report published by the daily analysis workflow.
AI_REPORT_FILE = (
    ROOT_DIR / "Data" / "Signals" / "current_btc_signal.json"
)

# Separate file for the monitor's pending paper signal.
PENDING_SIGNAL_FILE = (
    ROOT_DIR / "Data" / "Signals" / "pending_btc_signal.json"
)

STATE_FILE = (
    ROOT_DIR / "Data" / "Signals" / "btc_monitor_state.json"
)

OKX_BASE_URL = os.getenv(
    "OKX_BASE_URL",
    "https://www.okx.com",
).rstrip("/")

OKX_SYMBOL = "BTC-USDT"
OKX_BAR = "5m"

OKX_PAGE_SIZE = 100
MAX_HISTORY_PAGES = 20

REQUEST_TIMEOUT = 20
REQUEST_DELAY_SECONDS = 0.15

REPORT_SCHEMA_VERSION = "2.0"

MARKET_ENTRY_MAX_DELAY_SECONDS = 600
FIVE_MINUTES_SECONDS = 300


# ============================================================
# TIME AND JSON HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def utc_iso(timestamp=None):
    if timestamp is None:
        return utc_now().isoformat()

    return datetime.fromtimestamp(
        timestamp,
        timezone.utc,
    ).isoformat()


def parse_timestamp(value, field_name):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} is invalid.")

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} is invalid.")

        return float(value)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is missing.")

    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.timestamp()

    except ValueError as exc:
        raise ValueError(
            f"{field_name} is not a valid timestamp."
        ) from exc


def atomic_write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary.replace(path)


def read_json(path, default=None):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_state():
    state = read_json(
        STATE_FILE,
        default={},
    )

    if not isinstance(state, dict):
        raise ValueError(
            "Monitor state must be a JSON object."
        )

    state.setdefault("version", 2)
    state.setdefault("last_processed_5m_timestamp", 0)
    state.setdefault("last_analysis_id", None)
    state.setdefault("last_trade_action_analysis_id", None)
    state.setdefault("pending_signal", None)
    state.setdefault("pending_ai_close", None)
    state.setdefault("notification_ids", [])
    state.setdefault("target_watch", None)

    return state


def save_state(state):
    atomic_write_json(
        STATE_FILE,
        state,
    )


def save_current_signal(signal):
    atomic_write_json(
        PENDING_SIGNAL_FILE,
        signal,
    )


# ============================================================
# TELEGRAM
# ============================================================

def make_notification_id(event, identifier):
    raw = f"{event}|{identifier}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def send_telegram(message, event_id, state):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print(
            "Telegram skipped: credentials are not configured."
        )
        return False

    sent_ids = state.setdefault(
        "notification_ids",
        [],
    )

    if event_id in sent_ids:
        return False

    url = (
        f"https://api.telegram.org/bot{token}/sendMessage"
    )

    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        if not result.get("ok"):
            print(
                "Telegram rejected notification."
            )
            return False

        sent_ids.append(event_id)
        state["notification_ids"] = sent_ids[-500:]

        save_state(state)

        return True

    except Exception as exc:
        print(
            f"Telegram notification failed: {exc}"
        )
        return False


# ============================================================
# OKX MARKET DATA
# ============================================================

def okx_request(endpoint, params):
    query = urllib.parse.urlencode(params)

    url = (
        f"{OKX_BASE_URL}{endpoint}"
        f"?{query}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BTC-AI-Paper-Monitor/2.0",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    if payload.get("code") != "0":
        raise RuntimeError(
            "OKX API error: "
            f"{payload.get('msg', 'Unknown error')}"
        )

    return payload.get("data", [])


def fetch_okx_page(before=None):
    params = {
        "instId": OKX_SYMBOL,
        "bar": OKX_BAR,
        "limit": str(OKX_PAGE_SIZE),
    }

    if before is not None:
        params["after"] = str(before)

    return okx_request(
        "/api/v5/market/history-candles",
        params,
    )


def normalize_okx_candle(raw):
    if not isinstance(raw, list) or len(raw) < 9:
        raise ValueError(
            "Unexpected OKX candle format."
        )

    if str(raw[8]) != "1":
        return None

    timestamp_ms = int(raw[0])

    candle = {
        "timestamp": timestamp_ms // 1000,
        "open": float(raw[1]),
        "high": float(raw[2]),
        "low": float(raw[3]),
        "close": float(raw[4]),
        "volume": float(raw[5]),
    }

    values = [
        candle["open"],
        candle["high"],
        candle["low"],
        candle["close"],
        candle["volume"],
    ]

    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "OKX returned non-finite candle values."
        )

    if min(values[:4]) <= 0 or candle["volume"] < 0:
        raise ValueError(
            "OKX returned invalid candle values."
        )

    if (
        candle["high"] < candle["low"]
        or candle["high"] < max(candle["open"], candle["close"])
        or candle["low"] > min(candle["open"], candle["close"])
    ):
        raise ValueError(
            "OKX returned invalid candle geometry."
        )

    return candle


def fetch_new_confirmed_candles(last_timestamp):
    """Recover missing candles from the last saved checkpoint."""

    collected = {}
    before = None
    reached_checkpoint = False

    for page_number in range(MAX_HISTORY_PAGES):
        page = fetch_okx_page(
            before=before,
        )

        if not page:
            break

        oldest_timestamp_ms = min(
            int(row[0])
            for row in page
        )

        for raw in page:
            candle = normalize_okx_candle(raw)

            if candle is None:
                continue

            timestamp = candle["timestamp"]

            if timestamp > last_timestamp:
                collected[timestamp] = candle

            elif timestamp <= last_timestamp:
                reached_checkpoint = True

        if reached_checkpoint:
            break

        before = oldest_timestamp_ms

        time.sleep(REQUEST_DELAY_SECONDS)

    if last_timestamp > 0 and not reached_checkpoint:
        raise RuntimeError(
            "Checkpoint recovery was incomplete. "
            "The monitor will not advance its checkpoint."
        )

    return [
        collected[timestamp]
        for timestamp in sorted(collected)
    ]


# ============================================================
# AI REPORT VALIDATION
# ============================================================

def load_ai_report():
    report = read_json(AI_REPORT_FILE)

    if not isinstance(report, dict):
        raise ValueError(
            "AI report is missing or invalid."
        )

    if str(report.get("schema_version")) != REPORT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported AI report schema version."
        )

    if report.get("symbol") not in (
        "BTC-USDT",
        "BTC_USDT",
    ):
        raise ValueError(
            "AI report symbol must be BTC-USDT."
        )

    analysis_id = report.get("analysis_id")

    if (
        not isinstance(analysis_id, str)
        or not analysis_id.strip()
    ):
        raise ValueError(
            "AI report analysis_id is missing."
        )

    if report.get("decision") not in (
        "LONG",
        "SHORT",
        "WAIT",
    ):
        raise ValueError(
            "AI decision must be LONG, SHORT, or WAIT."
        )

    generated_at = parse_timestamp(
        report.get("generated_at"),
        "generated_at",
    )

    now = utc_now().timestamp()

    if generated_at > now + 60:
        raise ValueError(
            "AI report timestamp is in the future."
        )

    if generated_at < now - 86400:
        raise ValueError(
            "AI report is older than 24 hours."
        )

    confidence = report.get("confidence")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0 <= confidence <= 100
    ):
        raise ValueError(
            "AI confidence must be between 0 and 100."
        )

    return report, analysis_id, generated_at


def report_entry(report):
    entry = report.get("entry")

    if not isinstance(entry, dict):
        raise ValueError(
            "AI report entry must be an object."
        )

    entry_type = (
        entry.get("type")
        or entry.get("entry_type")
    )

    entry_price = (
        entry.get("price")
        if entry.get("price") is not None
        else entry.get("entry_price")
    )

    expires_at = (
        entry.get("expires_at")
        or entry.get("entry_expires_at")
        or report.get("entry_expires_at")
    )

    if entry_type not in (
        "MARKET",
        "LIMIT",
        "NONE",
    ):
        raise ValueError(
            "AI entry type must be MARKET, LIMIT, or NONE."
        )

    if entry_type == "NONE":
        return None

    if (
        isinstance(entry_price, bool)
        or not isinstance(entry_price, (int, float))
        or not math.isfinite(float(entry_price))
        or entry_price <= 0
    ):
        raise ValueError(
            "AI entry price must be a positive finite number."
        )

    zone_low = entry.get("entry_zone_low")
    zone_high = entry.get("entry_zone_high")

    if zone_low is not None or zone_high is not None:
        if (
            isinstance(zone_low, bool)
            or isinstance(zone_high, bool)
            or not isinstance(zone_low, (int, float))
            or not isinstance(zone_high, (int, float))
            or not math.isfinite(float(zone_low))
            or not math.isfinite(float(zone_high))
            or zone_low <= 0
            or zone_high <= 0
            or zone_low > zone_high
        ):
            raise ValueError(
                "AI entry zone is invalid."
            )

        if not zone_low <= entry_price <= zone_high:
            raise ValueError(
                "AI entry price is outside its entry zone."
            )

    return {
        "type": entry_type,
        "price": float(entry_price),
        "zone": {
            "low": zone_low,
            "high": zone_high,
        },
        "expires_at": expires_at,
    }


def report_levels(report):
    aliases = {
        "stop_loss": ("stop_loss", "sl"),
        "take_profit_1": ("take_profit_1", "tp1"),
        "take_profit_2": ("take_profit_2", "tp2"),
        "take_profit_3": ("take_profit_3", "tp3"),
    }

    levels = {}

    for output_name, candidates in aliases.items():
        value = None

        for candidate in candidates:
            if report.get(candidate) is not None:
                value = report[candidate]
                break

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(
                f"AI report has an invalid {output_name}."
            )

        levels[output_name] = float(value)

    return levels


def validate_ai_plan(report):
    decision = report["decision"]

    if decision == "WAIT":
        entry = report_entry(report)

        if entry is not None:
            raise ValueError(
                "WAIT decision must use entry_type NONE."
            )

        for field in (
            "stop_loss",
            "take_profit_1",
            "take_profit_2",
            "take_profit_3",
        ):
            if report.get(field) is not None:
                raise ValueError(
                    "WAIT decision must not contain trade levels."
                )

        return None

    entry = report_entry(report)

    if entry is None:
        raise ValueError(
            "LONG/SHORT requires a MARKET or LIMIT entry."
        )

    levels = report_levels(report)

    paper_trading._validate_trade_levels(
        decision,
        entry["price"],
        levels["stop_loss"],
        levels["take_profit_1"],
        levels["take_profit_2"],
        levels["take_profit_3"],
    )

    if entry["type"] == "LIMIT":
        if not entry.get("expires_at"):
            raise ValueError(
                "LIMIT entry requires an expiry timestamp."
            )

        expiry = parse_timestamp(
            entry["expires_at"],
            "entry.expires_at",
        )

        if expiry <= parse_timestamp(
            report["generated_at"],
            "generated_at",
        ):
            raise ValueError(
                "LIMIT expiry must be after report generation."
            )

    return {
        "direction": decision,
        "entry": entry,
        "levels": levels,
    }


# ============================================================
# SIGNAL MANAGEMENT
# ============================================================

def build_pending_signal(report, analysis_id, generated_at):
    plan = validate_ai_plan(report)

    if plan is None:
        return None

    return {
        "analysis_id": analysis_id,
        "generated_at": report["generated_at"],
        "generated_at_timestamp": generated_at,
        "symbol": "BTC-USDT",
        "direction": plan["direction"],
        "entry": plan["entry"],
        "levels": plan["levels"],
        "confidence": report.get("confidence"),
        "market_bias": report.get("market_bias"),
        "reasoning": report.get("reasoning"),
        "status": "PENDING",
        "created_at": utc_now().isoformat(),
    }


def format_signal_message(signal):
    entry = signal["entry"]
    levels = signal["levels"]

    reasoning = signal.get("reasoning")

    if isinstance(reasoning, (dict, list)):
        reasoning_text = json.dumps(
            reasoning,
            ensure_ascii=False,
        )
    else:
        reasoning_text = str(
            reasoning or "Not provided."
        )

    return (
        "BTC AI PAPER SIGNAL\n"
        f"Decision: {signal['direction']}\n"
        f"Entry type: {entry['type']}\n"
        f"Entry price: {entry['price']}\n"
        f"Stop loss: {levels['stop_loss']}\n"
        f"TP1: {levels['take_profit_1']}\n"
        f"TP2: {levels['take_profit_2']}\n"
        f"TP3: {levels['take_profit_3']}\n"
        f"Confidence: {signal.get('confidence')}\n"
        f"Reasoning: {reasoning_text}\n"
        "Paper trading only. No exchange order submitted."
    )


def notify_signal(event, signal, state):
    event_id = make_notification_id(
        event,
        signal["analysis_id"],
    )

    send_telegram(
        format_signal_message(signal),
        event_id,
        state,
    )


def get_open_trades(ledger):
    trades = ledger.get("trades")

    if not isinstance(trades, list):
        raise ValueError(
            "Paper trading ledger has an invalid trades list."
        )

    return [
        trade
        for trade in trades
        if trade.get("status") == "OPEN"
    ]


def get_updated_levels(report):
    levels = report.get(
        "open_trade_updated_levels"
    )

    if not isinstance(levels, dict):
        raise ValueError(
            "AI updated levels must be an object."
        )

    allowed = {
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "take_profit_3",
    }

    if set(levels) - allowed:
        raise ValueError(
            "AI supplied unsupported updated level fields."
        )

    if not any(levels.get(key) is not None for key in allowed):
        raise ValueError(
            "AI UPDATE_LEVELS must change at least one level."
        )

    for key in allowed:
        value = levels.get(key)

        if value is None:
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(
                f"Invalid updated trade level: {key}."
            )

    return levels


def process_new_ai_report(
    report,
    analysis_id,
    generated_at,
    state,
    ledger,
):
    """Validate the complete AI report before changing persistent state."""

    if state.get("last_analysis_id") == analysis_id:
        return

    # --------------------------------------------------------
    # 1. Validate the new AI decision.
    # --------------------------------------------------------

    plan = validate_ai_plan(report)

    previous_action = report.get(
        "previous_signal_action"
    )

    if previous_action not in (
        "NONE",
        "KEEP",
        "REPLACE",
        "CANCEL",
    ):
        raise ValueError(
            "Invalid previous_signal_action."
        )

    open_action = report.get(
        "open_trade_action"
    )

    if open_action not in (
        "NONE",
        "HOLD",
        "CLOSE",
        "UPDATE_LEVELS",
    ):
        raise ValueError(
            "Invalid open_trade_action."
        )

    open_trades = get_open_trades(ledger)

    if not open_trades:
        if open_action != "NONE":
            raise ValueError(
                "AI requested an open-trade action, "
                "but there is no open trade."
            )

    elif open_action == "NONE":
        raise ValueError(
            "AI report must specify an action "
            "for the existing open trade."
        )

    # --------------------------------------------------------
    # 2. Prepare signal changes without modifying state.
    # --------------------------------------------------------

    current_pending = state.get(
        "pending_signal"
    )

    signal_changed = False
    new_signal = current_pending

    if previous_action == "CANCEL":
        new_signal = None
        signal_changed = current_pending is not None

    elif previous_action == "REPLACE":
        new_signal = build_pending_signal(
            report,
            analysis_id,
            generated_at,
        )
        signal_changed = True

    elif previous_action == "NONE" and current_pending is None:
        new_signal = build_pending_signal(
            report,
            analysis_id,
            generated_at,
        )
        signal_changed = new_signal is not None

    # --------------------------------------------------------
    # 3. Validate open trade updates on temporary copies.
    # --------------------------------------------------------

    levels = None
    trial_trade = None

    if open_action == "UPDATE_LEVELS":
        if len(open_trades) != 1:
            raise ValueError(
                "UPDATE_LEVELS requires exactly one open trade."
            )

        levels = get_updated_levels(report)
        trial_trade = copy.deepcopy(open_trades[0])

        paper_trading.update_trade_levels(
            trial_trade,
            stop_loss=levels.get("stop_loss"),
            take_profit_1=levels.get("take_profit_1"),
            take_profit_2=levels.get("take_profit_2"),
            take_profit_3=levels.get("take_profit_3"),
        )

    # --------------------------------------------------------
    # 4. Prepare close request without applying it yet.
    # --------------------------------------------------------

    close_request = None

    if open_action == "CLOSE":
        close_request = {
            "analysis_id": analysis_id,
            "generated_at_timestamp": generated_at,
            "reason": "AI_CLOSE",
            "trade_ids": [
                trade["trade_id"]
                for trade in open_trades
            ],
        }

    # --------------------------------------------------------
    # 5. All validation passed. Apply changes.
    # --------------------------------------------------------

    if open_action == "UPDATE_LEVELS":
        paper_trading.update_trade_levels(
            open_trades[0],
            stop_loss=levels.get("stop_loss"),
            take_profit_1=levels.get("take_profit_1"),
            take_profit_2=levels.get("take_profit_2"),
            take_profit_3=levels.get("take_profit_3"),
        )

        paper_trading.save_ledger(ledger)

    if signal_changed:
        state["pending_signal"] = new_signal
        save_current_signal(new_signal)

    if open_action == "CLOSE":
        state["pending_ai_close"] = close_request

    elif open_action == "HOLD":
        state["pending_ai_close"] = None

    state["last_analysis_id"] = analysis_id
    state["last_trade_action_analysis_id"] = analysis_id

    save_state(state)

    # --------------------------------------------------------
    # 6. Send notifications after validation and persistence.
    # --------------------------------------------------------

    if previous_action == "CANCEL" and signal_changed:
        send_telegram(
            "BTC AI signal cancelled.\n"
            f"Analysis ID: {analysis_id}",
            make_notification_id(
                "signal_cancelled",
                analysis_id,
            ),
            state,
        )

    elif signal_changed and new_signal is not None:
        notify_signal(
            "signal_replaced" if previous_action == "REPLACE"
            else "signal_created",
            new_signal,
            state,
        )

    if open_action == "UPDATE_LEVELS":
        send_telegram(
            "BTC PAPER TRADE LEVELS UPDATED BY AI\n"
            f"Trade ID: {open_trades[0]['trade_id']}",
            make_notification_id(
                "levels_updated",
                analysis_id,
            ),
            state,
        )

    elif open_action == "CLOSE":
        send_telegram(
            "BTC AI requested a paper trade close.\n"
            "The close will be processed on a confirmed "
            "5M candle after the AI report timestamp.\n"
            f"Analysis ID: {analysis_id}",
            make_notification_id(
                "close_requested",
                analysis_id,
            ),
            state,
        )


# ============================================================
# PAPER ENTRY EXECUTION
# ============================================================

def expire_pending_signal(state, reason):
    signal = state.get("pending_signal")

    if signal is None:
        return

    signal["status"] = "EXPIRED"
    signal["expiry_reason"] = reason
    signal["expired_at"] = utc_now().isoformat()

    state["last_expired_signal"] = signal
    state["pending_signal"] = None

    save_current_signal(None)
    save_state(state)

    send_telegram(
        "BTC AI paper signal expired.\n"
        f"Reason: {reason}\n"
        f"Analysis ID: {signal.get('analysis_id')}",
        make_notification_id(
            "signal_expired",
            signal.get("analysis_id"),
        ),
        state,
    )


def determine_entry_fill(signal, candle, state):
    entry = signal["entry"]
    entry_type = entry["type"]

    generated_at = signal["generated_at_timestamp"]
    candle_timestamp = candle["timestamp"]

    if candle_timestamp <= generated_at:
        return None

    if entry_type == "MARKET":
        delay = candle_timestamp - generated_at

        if delay > MARKET_ENTRY_MAX_DELAY_SECONDS:
            expire_pending_signal(
                state,
                "MARKET_ENTRY_WINDOW_MISSED",
            )
            return None

        return candle["open"]

    if entry_type == "LIMIT":
        expiry = parse_timestamp(
            entry["expires_at"],
            "entry.expires_at",
        )

        candle_close_timestamp = (
            candle_timestamp + FIVE_MINUTES_SECONDS
        )

        if candle_close_timestamp > expiry:
            expire_pending_signal(
                state,
                "LIMIT_ENTRY_EXPIRED",
            )
            return None

        limit_price = entry["price"]

        if not (
            candle["low"] <= limit_price <= candle["high"]
        ):
            return None

        direction = signal["direction"]

        if direction == "LONG" and candle["open"] <= limit_price:
            return candle["open"]

        if direction == "SHORT" and candle["open"] >= limit_price:
            return candle["open"]

        return limit_price

    raise ValueError(
        "Unsupported AI entry type."
    )


def try_open_pending_signal(candle, state, ledger):
    signal = state.get("pending_signal")

    if not signal:
        return False

    if signal.get("status") != "PENDING":
        return False

    # Do not open a second position while another trade is open.
    if get_open_trades(ledger):
        return False

    reference_price = determine_entry_fill(
        signal,
        candle,
        state,
    )

    if reference_price is None:
        return False

    entry = signal["entry"]
    levels = signal["levels"]

    # Revalidate the AI levels against the actual execution reference.
    try:
        paper_trading._validate_trade_levels(
            signal["direction"],
            reference_price,
            levels["stop_loss"],
            levels["take_profit_1"],
            levels["take_profit_2"],
            levels["take_profit_3"],
        )
    except ValueError:
        expire_pending_signal(
            state,
            "EXECUTION_PRICE_INVALIDATED_AI_LEVELS",
        )
        return False

    fingerprint = paper_trading.make_signal_fingerprint(
        signal["analysis_id"],
        signal["direction"],
        entry["type"],
        entry["price"],
    )

    trade, created = paper_trading.open_trade_from_ai(
        ledger=ledger,
        direction=signal["direction"],
        entry_type=entry["type"],
        entry_price=reference_price,
        stop_loss=levels["stop_loss"],
        take_profit_1=levels["take_profit_1"],
        take_profit_2=levels["take_profit_2"],
        take_profit_3=levels["take_profit_3"],
        signal_fingerprint=fingerprint,
        timestamp=candle["timestamp"] + FIVE_MINUTES_SECONDS,
    )

    if not created:
        signal["status"] = "ALREADY_PROCESSED"
        state["pending_signal"] = None

        save_current_signal(None)
        save_state(state)

        return False

    signal["status"] = "OPENED"
    signal["trade_id"] = trade["trade_id"]
    signal["opened_at"] = utc_iso(
        candle["timestamp"] + FIVE_MINUTES_SECONDS
    )

    state["target_watch"] = build_target_watch(
        signal,
        trade,
    )

    state["last_opened_analysis_id"] = signal["analysis_id"]
    state["pending_signal"] = None

    save_current_signal(None)

    paper_trading.save_ledger(ledger)
    save_state(state)

    send_telegram(
        "BTC PAPER TRADE OPENED\n"
        f"Direction: {trade['direction']}\n"
        f"Entry type: {trade['entry_type']}\n"
        f"Entry reference: {trade['entry_reference_price']}\n"
        f"Entry fill: {trade['entry_fill_price']}\n"
        f"Stop loss: {trade['stop_loss_current']}\n"
        f"TP1: {trade['take_profit_1']}\n"
        f"Trade ID: {trade['trade_id']}\n"
        "No exchange order submitted.",
        make_notification_id(
            "trade_opened",
            trade["trade_id"],
        ),
        state,
    )

    return True


# ============================================================
# OPEN TRADE MANAGEMENT
# ============================================================

def notify_closed_trades(
    ledger,
    previous_status,
    state,
):
    for trade in ledger["trades"]:
        trade_id = trade["trade_id"]

        if previous_status.get(trade_id) != "OPEN":
            continue

        if trade.get("status") != "CLOSED":
            continue

        send_telegram(
            "BTC PAPER TRADE CLOSED\n"
            f"Direction: {trade['direction']}\n"
            f"Reason: {trade['exit_reason']}\n"
            f"Exit reference: {trade['exit_price']}\n"
            f"Exit fill: {trade['exit_fill_price']}\n"
            f"Net PnL: {trade['realized_pnl_usd']:.2f} USD\n"
            f"Trade ID: {trade_id}",
            make_notification_id(
                "trade_closed",
                trade_id,
            ),
            state,
        )


def process_ai_close(candle, state, ledger):
    request = state.get(
        "pending_ai_close"
    )

    if not request:
        return

    candle_close_timestamp = (
        candle["timestamp"] + FIVE_MINUTES_SECONDS
    )

    generated_at = request.get(
        "generated_at_timestamp",
        0,
    )

    # Never apply a new AI instruction to an older candle.
    if candle_close_timestamp <= generated_at:
        return

    requested_ids = set(
        request.get("trade_ids", [])
    )

    open_trades = [
        trade
        for trade in get_open_trades(ledger)
        if (
            not requested_ids
            or trade["trade_id"] in requested_ids
        )
    ]

    previous_status = {
        trade["trade_id"]: "OPEN"
        for trade in open_trades
    }

    for trade in open_trades:
        paper_trading.close_trade_at_market(
            trade,
            reference_price=candle["close"],
            timestamp=candle_close_timestamp,
            reason="AI_CLOSE",
        )

    paper_trading.save_ledger(ledger)

    notify_closed_trades(
        ledger,
        previous_status,
        state,
    )

    state["pending_ai_close"] = None
    state["last_completed_ai_close_id"] = request["analysis_id"]

    save_state(state)



# ============================================================
# DAILY TARGET WATCH
# ============================================================

def next_baku_0800_timestamp(reference_timestamp):
    from zoneinfo import ZoneInfo
    from datetime import timedelta

    baku = ZoneInfo("Asia/Baku")
    local_time = datetime.fromtimestamp(
        reference_timestamp,
        timezone.utc,
    ).astimezone(baku)
    next_run = local_time.replace(
        hour=8,
        minute=0,
        second=0,
        microsecond=0,
    )
    if local_time >= next_run:
        next_run += timedelta(days=1)
    return next_run.astimezone(timezone.utc).timestamp()


def build_target_watch(signal, trade):
    entry_timestamp = int(trade["entry_timestamp"])
    return {
        "analysis_id": signal["analysis_id"],
        "trade_id": trade["trade_id"],
        "direction": trade["direction"],
        "entry_price": float(trade["entry_reference_price"]),
        "entry_timestamp": entry_timestamp,
        "stop_loss": float(trade["stop_loss_initial"]),
        "targets": {
            "TP1": float(trade["take_profit_1"]),
            "TP2": float(trade["take_profit_2"]),
            "TP3": float(trade["take_profit_3"]),
        },
        "reached": {
            "TP1": False,
            "TP2": False,
            "TP3": False,
        },
        "notification_sent": {
            "TP1": False,
            "TP2": False,
            "TP3": False,
            "STOP_LOSS": False,
        },
        "status": "ACTIVE",
        "expires_at": next_baku_0800_timestamp(entry_timestamp),
        "created_at": utc_iso(),
    }


def _target_touched(direction, high, low, level, target_name):
    if target_name == "STOP_LOSS":
        return low <= level if direction == "LONG" else high >= level
    return high >= level if direction == "LONG" else low <= level


def process_target_watch(candle, state):
    """Track daily plan milestones independently of paper position closure."""
    watch = state.get("target_watch")
    if not isinstance(watch, dict) or watch.get("status") != "ACTIVE":
        return

    candle_close_timestamp = int(candle["timestamp"]) + FIVE_MINUTES_SECONDS
    if candle_close_timestamp <= int(watch["entry_timestamp"]):
        return

    if candle_close_timestamp >= float(watch["expires_at"]):
        watch["status"] = "EXPIRED"
        watch["expired_at"] = utc_iso(candle_close_timestamp)
        save_state(state)
        send_telegram(
            "BTC DAILY TARGET WATCH EXPIRED\n"
            f"Analysis ID: {watch['analysis_id']}\n"
            "The daily monitoring window ended at 08:00 Baku.",
            make_notification_id("target_watch_expired", watch["analysis_id"]),
            state,
        )
        return

    high = float(candle["high"])
    low = float(candle["low"])
    direction = watch["direction"]

    # Before TP1, the original stop invalidates the target plan.
    # If stop and TP1 are both touched in one OHLC candle, stop is assumed first.
    if not watch["reached"]["TP1"] and _target_touched(
        direction, high, low, float(watch["stop_loss"]), "STOP_LOSS"
    ):
        watch["status"] = "INVALIDATED"
        watch["invalidated_at"] = utc_iso(candle_close_timestamp)
        watch["notification_sent"]["STOP_LOSS"] = False
        save_state(state)
        sent = send_telegram(
            "BTC DAILY PLAN INVALIDATED — STOP LOSS TOUCHED\n"
            f"Direction: {direction}\n"
            f"Stop loss: {watch['stop_loss']}\n"
            f"Analysis ID: {watch['analysis_id']}\n"
            "No further targets will be tracked for this plan.",
            make_notification_id("target_watch_stop", watch["analysis_id"]),
            state,
        )
        if sent:
            watch["notification_sent"]["STOP_LOSS"] = True
            save_state(state)
        return

    newly_reached = []
    for target_name in ("TP1", "TP2", "TP3"):
        if watch["reached"].get(target_name):
            continue
        target_price = float(watch["targets"][target_name])
        if _target_touched(direction, high, low, target_price, target_name):
            watch["reached"][target_name] = True
            watch["reached_at"] = watch.get("reached_at", {})
            watch["reached_at"][target_name] = utc_iso(candle_close_timestamp)
            newly_reached.append(target_name)

    if not newly_reached:
        return

    save_state(state)
    multiple = len(newly_reached) > 1
    for target_name in newly_reached:
        if watch["notification_sent"].get(target_name):
            continue
        note = (
            "\\nNote: one 5M candle crossed multiple targets; "
            "the exact intrabar order cannot be determined from OHLC data."
            if multiple else ""
        )
        message = (
            f"BTC {target_name} REACHED\\n"
            f"Direction: {direction}\\n"
            f"Target: {watch['targets'][target_name']}\\n"
            f"Analysis ID: {watch['analysis_id']}\\n"
            "Daily target monitoring remains active until the next 08:00 Baku analysis."
            f"{note}"
        )
        sent = send_telegram(
            message,
            make_notification_id(
                "daily_target_reached",
                f"{watch['analysis_id']}:{target_name}",
            ),
            state,
        )
        if sent:
            watch["notification_sent"][target_name] = True
            save_state(state)

    if all(watch["reached"].values()):
        watch["status"] = "TARGETS_COMPLETED"
        watch["completed_at"] = utc_iso(candle_close_timestamp)
        save_state(state)



def process_one_candle(candle, state, ledger):
    process_target_watch(candle, state)

    previous_status = {
        trade["trade_id"]: trade.get("status")
        for trade in ledger["trades"]
    }

    # Manage existing positions first.
    paper_trading.process_candle(
        ledger,
        candle,
    )

    paper_trading.save_ledger(ledger)

    notify_closed_trades(
        ledger,
        previous_status,
        state,
    )

    # Apply an AI close only after its report timestamp.
    process_ai_close(
        candle,
        state,
        ledger,
    )

    # Only consider a new entry when no position is open.
    try_open_pending_signal(
        candle,
        state,
        ledger,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting BTC AI paper-trading monitor.")

    state = load_state()
    ledger = paper_trading.load_ledger()

    # AI report is mandatory.
    if not AI_REPORT_FILE.is_file():
        raise FileNotFoundError(
            f"AI report not found: {AI_REPORT_FILE}"
        )

    report, analysis_id, generated_at = load_ai_report()

    print(f"AI analysis ID: {analysis_id}")
    print(f"AI decision: {report['decision']}")

    process_new_ai_report(
        report,
        analysis_id,
        generated_at,
        state,
        ledger,
    )

    last_timestamp = int(
        state.get(
            "last_processed_5m_timestamp",
            0,
        )
    )

    candles = fetch_new_confirmed_candles(
        last_timestamp
    )

    if not candles:
        print("No new confirmed 5M candles.")

        result = paper_trading.summary(ledger)

        print(f"Total trades: {result['total_trades']}")
        print(f"Open trades: {result['open_trades']}")
        print(f"Closed trades: {result['closed_trades']}")

        print(
            "Realized net PnL: "
            f"{result['realized_net_pnl_usd']:.2f} USD"
        )

        print(
            "Unrealized net PnL: "
            f"{result['unrealized_net_pnl_usd']:.2f} USD"
        )

        print(
            "Combined net PnL: "
            f"{result['combined_net_pnl_usd']:.2f} USD"
        )

        return

    processed_count = 0

    for candle in candles:
        if candle["timestamp"] <= last_timestamp:
            continue

        process_one_candle(
            candle,
            state,
            ledger,
        )

        last_timestamp = candle["timestamp"]

        state["last_processed_5m_timestamp"] = (
            last_timestamp
        )

        save_state(state)

        processed_count += 1

    paper_trading.save_ledger(ledger)

    result = paper_trading.summary(ledger)

    print(f"Candles processed: {processed_count}")
    print(f"Total trades: {result['total_trades']}")
    print(f"Open trades: {result['open_trades']}")
    print(f"Closed trades: {result['closed_trades']}")

    print(
        "Realized net PnL: "
        f"{result['realized_net_pnl_usd']:.2f} USD"
    )

    print(
        "Unrealized net PnL: "
        f"{result['unrealized_net_pnl_usd']:.2f} USD"
    )

    print(
        "Combined net PnL: "
        f"{result['combined_net_pnl_usd']:.2f} USD"
    )


if __name__ == "__main__":
    main()
