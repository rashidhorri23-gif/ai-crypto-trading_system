
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SIGNAL_FILE = ROOT / "Data" / "Signals" / "current_btc_signal.json"
STATE_FILE = ROOT / "Data" / "Signals" / "btc_monitor_state.json"

OKX_BASE_URL = "https://www.okx.com"
INSTRUMENT = "BTC-USDT"

FOUR_HOURS_SECONDS = 4 * 60 * 60
FIVE_MINUTES_SECONDS = 5 * 60

DIRECTIONS = {
    "LONG": {
        "plan_key": "bullish_plan",
        "trigger_type": "above",
    },
    "SHORT": {
        "plan_key": "bearish_plan",
        "trigger_type": "below",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{utc_now()}] {message}", flush=True)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    temporary_path.replace(path)


def fetch_candles(bar, limit=100):
    params = urllib.parse.urlencode({
        "instId": INSTRUMENT,
        "bar": bar,
        "limit": str(limit),
    })

    url = f"{OKX_BASE_URL}/api/v5/market/candles?{params}"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BTC-Signal-Monitor/1.0"},
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("code") != "0":
        raise RuntimeError(f"OKX API error: {payload}")

    candles = []

    for row in payload.get("data", []):
        if len(row) < 9:
            continue

        candle = {
            "timestamp": int(row[0]) // 1000,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "confirm": str(row[8]),
        }

        if candle["confirm"] == "1":
            candles.append(candle)

    candles.sort(key=lambda item: item["timestamp"])

    if not candles:
        raise RuntimeError(f"No confirmed candles returned for {bar}")

    return candles


def make_fingerprint(direction, plan):
    fields = [
        direction,
        plan.get("setup_type"),
        plan.get("entry_zone_low"),
        plan.get("entry_zone_high"),
        plan.get("stop_loss"),
        plan.get("take_profit_1"),
        plan.get("take_profit_2"),
        plan.get("take_profit_3"),
    ]

    raw = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_plan(plan):
    required = [
        "entry_zone_low",
        "entry_zone_high",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "take_profit_3",
    ]

    for field in required:
        value = plan.get(field)

        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Invalid plan field: {field}")

    if plan["entry_zone_low"] > plan["entry_zone_high"]:
        raise ValueError("Invalid entry zone")

    return True


def load_state():
    if not STATE_FILE.exists():
        return {
            "version": 1,
            "scenarios": {},
            "sent_fingerprints": [],
        }

    state = load_json(STATE_FILE)

    state.setdefault("scenarios", {})
    state.setdefault("sent_fingerprints", [])

    return state


def initialize_scenario(direction, plan, latest_4h, state):
    fingerprint = make_fingerprint(direction, plan)

    previous = state["scenarios"].get(direction)

    if previous and previous.get("fingerprint") == fingerprint:
        return previous

    already_sent = fingerprint in state["sent_fingerprints"]

    scenario = {
        "fingerprint": fingerprint,
        "baseline_4h_timestamp": latest_4h["timestamp"],
        "breakout_timestamp": None,
        "alert_sent": already_sent,
        "invalidated": False,
        "created_at": utc_now(),
    }

    state["scenarios"][direction] = scenario

    log(
        f"{direction} scenario initialized. "
        f"Waiting for a new confirmed 4H breakout candle."
    )

    return scenario


def find_breakout(direction, plan, scenario, candles_4h):
    if scenario.get("breakout_timestamp") is not None:
        return scenario["breakout_timestamp"]

    baseline = scenario["baseline_4h_timestamp"]
    trigger_level = float(plan["entry_zone_low"])

    for candle in candles_4h:
        if candle["timestamp"] <= baseline:
            continue

        if direction == "LONG" and candle["close"] > trigger_level:
            breakout_close_time = (
                candle["timestamp"] + FOUR_HOURS_SECONDS
            )

            scenario["breakout_timestamp"] = breakout_close_time

            log(
                f"LONG breakout confirmed at "
                f"{candle['close']:.2f}"
            )

            return breakout_close_time

        if direction == "SHORT" and candle["close"] < trigger_level:
            breakout_close_time = (
                candle["timestamp"] + FOUR_HOURS_SECONDS
            )

            scenario["breakout_timestamp"] = breakout_close_time

            log(
                f"SHORT breakdown confirmed at "
                f"{candle['close']:.2f}"
            )

            return breakout_close_time

    return None


def check_invalidation(direction, plan, scenario, latest_4h):
    if scenario.get("breakout_timestamp") is None:
        return False

    stop_loss = float(plan["stop_loss"])
    close = latest_4h["close"]

    if direction == "LONG" and close < stop_loss:
        return True

    if direction == "SHORT" and close > stop_loss:
        return True

    return False


def find_retest(direction, plan, breakout_timestamp, candles_5m):
    zone_low = float(plan["entry_zone_low"])
    zone_high = float(plan["entry_zone_high"])

    for candle in candles_5m:
        if candle["timestamp"] < breakout_timestamp:
            continue

        overlaps_zone = (
            candle["low"] <= zone_high
            and candle["high"] >= zone_low
        )

        if not overlaps_zone:
            continue

        if direction == "LONG":
            bullish_close = (
                candle["close"] > candle["open"]
                and candle["close"] > zone_low
            )

            if bullish_close:
                return candle

        if direction == "SHORT":
            bearish_close = (
                candle["close"] < candle["open"]
                and candle["close"] < zone_high
            )

            if bearish_close:
                return candle

    return None


def build_alert(direction, plan, candle):
    entry_low = float(plan["entry_zone_low"])
    entry_high = float(plan["entry_zone_high"])

    candle_time = datetime.fromtimestamp(
        candle["timestamp"] + FIVE_MINUTES_SECONDS,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"BTC {direction} SIGNAL CONFIRMED\n\n"
        f"Timeframe: 4H + 5M\n"
        f"Confirmation: {candle_time}\n"
        f"Entry Zone: {entry_low:.2f} - {entry_high:.2f}\n"
        f"Stop Loss: {float(plan['stop_loss']):.2f}\n"
        f"Take Profit 1: {float(plan['take_profit_1']):.2f}\n"
        f"Take Profit 2: {float(plan['take_profit_2']):.2f}\n"
        f"Take Profit 3: {float(plan['take_profit_3']):.2f}\n\n"
        f"Confirmation Candle Close: {candle['close']:.2f}\n"
        f"Source: OKX BTC-USDT\n\n"
        f"Manual execution only. No automatic trade was opened."
    )


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        log("Telegram credentials are missing. Alert remains pending.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    body = json.dumps({
        "chat_id": chat_id,
        "text": message,
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")

    return True


def process_scenario(direction, plan, state, candles_4h, candles_5m):
    validate_plan(plan)

    scenario = initialize_scenario(
        direction,
        plan,
        candles_4h[-1],
        state,
    )

    if scenario["alert_sent"]:
        log(f"{direction}: alert already sent.")
        return

    if scenario["invalidated"]:
        log(f"{direction}: scenario is invalidated.")
        return

    breakout_timestamp = find_breakout(
        direction,
        plan,
        scenario,
        candles_4h,
    )

    if breakout_timestamp is None:
        log(f"{direction}: waiting for 4H breakout.")
        return

    if check_invalidation(
        direction,
        plan,
        scenario,
        candles_4h[-1],
    ):
        scenario["invalidated"] = True
        log(f"{direction}: scenario invalidated by 4H close.")
        return

    retest = find_retest(
        direction,
        plan,
        breakout_timestamp,
        candles_5m,
    )

    if retest is None:
        log(f"{direction}: breakout confirmed; waiting for 5M retest.")
        return

    message = build_alert(direction, plan, retest)

    log(message)

    if send_telegram(message):
        scenario["alert_sent"] = True

        if scenario["fingerprint"] not in state["sent_fingerprints"]:
            state["sent_fingerprints"].append(scenario["fingerprint"])

        log(f"{direction}: Telegram alert sent successfully.")


def main():
    log("Starting BTC signal monitor.")

    signal = load_json(SIGNAL_FILE)
    state = load_state()

    candles_4h = fetch_candles("4H", limit=100)
    candles_5m = fetch_candles("5m", limit=300)

    log(
        f"Loaded {len(candles_4h)} confirmed 4H candles "
        f"and {len(candles_5m)} confirmed 5M candles."
    )

    for direction, settings in DIRECTIONS.items():
        plan = signal.get(settings["plan_key"])

        if not isinstance(plan, dict):
            log(f"{direction}: plan is missing.")
            continue

        process_scenario(
            direction,
            plan,
            state,
            candles_4h,
            candles_5m,
        )

    save_json(STATE_FILE, state)

    log("BTC signal monitoring completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"MONITOR ERROR: {error}")
        sys.exit(1)
