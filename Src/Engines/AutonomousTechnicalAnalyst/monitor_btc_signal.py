
import hashlib
import json
import math
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

CANDLE_LIMIT_4H = 300
CANDLE_LIMIT_5M = 300

MAX_HISTORY_PAGES = 20
MONITOR_LOGIC_VERSION = 2

DIRECTIONS = {
    "LONG": {"plan_key": "bullish_plan"},
    "SHORT": {"plan_key": "bearish_plan"},
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


def parse_candle_rows(rows):
    candles = []

    for row in rows:
        if len(row) < 9:
            continue

        try:
            candle = {
                "timestamp": int(row[0]) // 1000,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "confirm": str(row[8]),
            }
        except (TypeError, ValueError):
            continue

        if candle["confirm"] != "1":
            continue

        prices = [
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"],
        ]

        if not all(math.isfinite(p) and p > 0 for p in prices):
            continue

        if candle["high"] < max(
            candle["open"],
            candle["close"],
            candle["low"],
        ):
            continue

        if candle["low"] > min(
            candle["open"],
            candle["close"],
            candle["high"],
        ):
            continue

        candles.append(candle)

    unique = {
        candle["timestamp"]: candle
        for candle in candles
    }

    return sorted(unique.values(), key=lambda item: item["timestamp"])


def request_candle_page(bar, limit=300, after=None, history=False):
    params = {
        "instId": INSTRUMENT,
        "bar": bar,
        "limit": str(limit),
    }

    if after is not None:
        params["after"] = str(int(after))

    endpoint = (
        "/api/v5/market/history-candles"
        if history
        else "/api/v5/market/candles"
    )

    url = (
        f"{OKX_BASE_URL}{endpoint}?"
        f"{urllib.parse.urlencode(params)}"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BTC-Signal-Monitor/1.0"},
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("code") != "0":
        raise RuntimeError(f"OKX API error: {payload}")

    return parse_candle_rows(payload.get("data", []))


def fetch_candles(bar, limit):
    candles = request_candle_page(
        bar=bar,
        limit=limit,
        history=False,
    )

    if not candles:
        raise RuntimeError(f"No confirmed candles returned for {bar}")

    return candles


def fetch_candles_since(bar, initial_candles, last_processed, interval):
    if not initial_candles:
        raise RuntimeError(
            "No initial candles available for history recovery."
        )

    candles_by_timestamp = {
        candle["timestamp"]: candle
        for candle in initial_candles
    }

    if len(candles_by_timestamp) != len(initial_candles):
        log("Duplicate candle timestamps detected and removed.")

    if last_processed is None:
        return sorted(
            candles_by_timestamp.values(),
            key=lambda item: item["timestamp"],
        )

    if isinstance(last_processed, bool) or not isinstance(
        last_processed, int
    ):
        raise RuntimeError("Invalid last processed timestamp.")

    def find_gaps(candles):
        ordered = sorted(
            candles,
            key=lambda item: item["timestamp"],
        )

        gaps = []

        for index in range(1, len(ordered)):
            previous_timestamp = ordered[index - 1]["timestamp"]
            current_timestamp = ordered[index]["timestamp"]

            difference = current_timestamp - previous_timestamp

            if difference != interval:
                if difference < interval or difference % interval != 0:
                    raise RuntimeError(
                        "Invalid candle timestamp sequence detected."
                    )

                gaps.append({
                    "start": previous_timestamp,
                    "end": current_timestamp,
                    "missing": difference // interval - 1,
                })

        return ordered, gaps

    pages = 0

    while True:
        ordered, gaps = find_gaps(
            list(candles_by_timestamp.values())
        )

        oldest_timestamp = ordered[0]["timestamp"]
        required_oldest = last_processed + interval

        if oldest_timestamp > required_oldest:
            if pages >= MAX_HISTORY_PAGES:
                raise RuntimeError(
                    "History recovery exceeded the page limit. "
                    "Checkpoint will not advance."
                )

            log(
                "Historical coverage gap detected. "
                "Recovering older candles from OKX."
            )

            older_candles = request_candle_page(
                bar=bar,
                limit=300,
                after=oldest_timestamp * 1000,
                history=True,
            )

            older_candles = [
                candle
                for candle in older_candles
                if candle["timestamp"] < oldest_timestamp
            ]

            if not older_candles:
                raise RuntimeError(
                    "OKX returned no older candles during recovery. "
                    "Checkpoint will not advance."
                )

            previous_oldest = oldest_timestamp

            for candle in older_candles:
                candles_by_timestamp[candle["timestamp"]] = candle

            new_oldest = min(candles_by_timestamp)

            if new_oldest >= previous_oldest:
                raise RuntimeError(
                    "History pagination made no progress."
                )

            pages += 1

            log(
                f"History page {pages} recovered. "
                f"Oldest candle: {new_oldest}"
            )

            continue

        if not gaps:
            break

        gap = gaps[0]

        log(
            "Internal candle gap detected. "
            f"Missing candles: {gap['missing']}"
        )

        if pages >= MAX_HISTORY_PAGES:
            raise RuntimeError(
                "History recovery exceeded the page limit "
                "while repairing an internal gap."
            )

        # OKX 'after' returns candles older than the supplied timestamp.
        # Use the end of the gap to retrieve candles inside that gap.
        recovered_page = request_candle_page(
            bar=bar,
            limit=300,
            after=gap["end"] * 1000,
            history=True,
        )

        recovered_page = [
            candle
            for candle in recovered_page
            if gap["start"] < candle["timestamp"] < gap["end"]
        ]

        if not recovered_page:
            raise RuntimeError(
                "OKX could not recover an internal candle gap. "
                "Checkpoint will not advance."
            )

        previous_count = len(candles_by_timestamp)

        for candle in recovered_page:
            candles_by_timestamp[candle["timestamp"]] = candle

        if len(candles_by_timestamp) == previous_count:
            raise RuntimeError(
                "Internal gap recovery made no progress."
            )

        pages += 1

    ordered, remaining_gaps = find_gaps(
        list(candles_by_timestamp.values())
    )

    if remaining_gaps:
        raise RuntimeError(
            "Candle history still contains unresolved gaps. "
            "Checkpoint will not advance."
        )

    if ordered[-1]["timestamp"] < last_processed:
        raise RuntimeError(
            "Latest available candle is older than the checkpoint."
        )

    log(
        f"History validation completed. "
        f"Total candles: {len(ordered)}. "
        f"Recovered pages: {pages}."
    )

    return ordered


def make_fingerprint(direction, plan):
    fields = [
        MONITOR_LOGIC_VERSION,
        direction,
        plan.get("setup_type"),
        plan.get("entry_zone_low"),
        plan.get("entry_zone_high"),
        plan.get("stop_loss"),
        plan.get("take_profit_1"),
        plan.get("take_profit_2"),
        plan.get("take_profit_3"),
        plan.get("activation_condition"),
        plan.get("invalidation_condition"),
    ]

    raw = json.dumps(fields, sort_keys=True)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def validate_plan(direction, plan):
    if direction not in DIRECTIONS:
        raise ValueError(f"Invalid direction: {direction}")

    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object.")

    required_fields = [
        "entry_zone_low",
        "entry_zone_high",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "take_profit_3",
    ]

    for field in required_fields:
        value = plan.get(field)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid numeric value: {field}")

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive finite value: {field}")

    low = float(plan["entry_zone_low"])
    high = float(plan["entry_zone_high"])
    stop = float(plan["stop_loss"])

    tp1 = float(plan["take_profit_1"])
    tp2 = float(plan["take_profit_2"])
    tp3 = float(plan["take_profit_3"])

    if low >= high:
        raise ValueError("Entry zone low must be below entry zone high.")

    if direction == "LONG":
        if not stop < low:
            raise ValueError("LONG stop-loss must be below the entry zone.")

        if not high < tp1 < tp2 < tp3:
            raise ValueError(
                "LONG targets must be above the entry zone and ascending."
            )

    elif direction == "SHORT":
        if not stop > high:
            raise ValueError("SHORT stop-loss must be above the entry zone.")

        if not low > tp1 > tp2 > tp3:
            raise ValueError(
                "SHORT targets must be below the entry zone and descending."
            )

    return True


def load_state():
    if not STATE_FILE.exists():
        return {
            "version": 1,
            "scenarios": {},
            "sent_fingerprints": [],
            "last_processed_5m_timestamp": None,
        }

    state = load_json(STATE_FILE)

    if not isinstance(state, dict):
        raise ValueError("Monitor state must be a JSON object.")

    state.setdefault("version", 1)
    state.setdefault("scenarios", {})
    state.setdefault("sent_fingerprints", [])
    state.setdefault("last_processed_5m_timestamp", None)

    if not isinstance(state["scenarios"], dict):
        raise ValueError("Invalid scenarios state.")

    if not isinstance(state["sent_fingerprints"], list):
        raise ValueError("Invalid sent fingerprints state.")

    return state


def initialize_scenario(direction, plan, latest_4h, state):
    fingerprint = make_fingerprint(direction, plan)

    previous = state["scenarios"].get(direction)

    if previous and previous.get("fingerprint") == fingerprint:
        previous.setdefault("sweep_timestamp", None)
        previous.setdefault("breakout_timestamp", None)
        previous.setdefault("pending_retest", None)
        previous.setdefault("alert_sent", False)
        previous.setdefault("invalidated", False)
        return previous

    already_sent = fingerprint in state["sent_fingerprints"]

    scenario = {
        "fingerprint": fingerprint,
        "baseline_4h_timestamp": latest_4h["timestamp"],
        "sweep_timestamp": None,
        "breakout_timestamp": None,
        "pending_retest": None,
        "alert_sent": already_sent,
        "invalidated": False,
        "created_at": utc_now(),
    }

    state["scenarios"][direction] = scenario

    log(
        f"{direction}: new scenario registered. "
        "Waiting for a new confirmed 4H activation."
    )

    return scenario


def find_long_activation(plan, scenario, candles_4h):
    existing_timestamp = scenario.get("breakout_timestamp")

    if existing_timestamp is not None:
        return existing_timestamp

    baseline = scenario["baseline_4h_timestamp"]

    zone_low = float(plan["entry_zone_low"])
    zone_high = float(plan["entry_zone_high"])
    stop_loss = float(plan["stop_loss"])

    for candle in candles_4h:
        timestamp = candle["timestamp"]

        if timestamp <= baseline:
            continue

        if scenario.get("sweep_timestamp") is None:
            if candle["low"] < zone_low:
                scenario["sweep_timestamp"] = timestamp

                log(
                    "LONG liquidity sweep detected. "
                    f"Low: {candle['low']:.2f}"
                )

                if candle["close"] < stop_loss:
                    scenario["invalidated"] = True
                    return "INVALIDATED"

            continue

        if timestamp <= scenario["sweep_timestamp"]:
            continue

        if candle["close"] < stop_loss:
            scenario["invalidated"] = True
            return "INVALIDATED"

        if candle["close"] > zone_high:
            breakout_close_time = timestamp + FOUR_HOURS_SECONDS

            scenario["breakout_timestamp"] = breakout_close_time

            log(
                "LONG reclaim confirmed after liquidity sweep. "
                f"4H close: {candle['close']:.2f}"
            )

            return breakout_close_time

    return None


def find_short_breakdown(plan, scenario, candles_4h):
    existing_timestamp = scenario.get("breakout_timestamp")

    if existing_timestamp is not None:
        return existing_timestamp

    baseline = scenario["baseline_4h_timestamp"]
    trigger_level = float(plan["entry_zone_low"])

    for candle in candles_4h:
        if candle["timestamp"] <= baseline:
            continue

        if candle["close"] < trigger_level:
            breakout_close_time = (
                candle["timestamp"] + FOUR_HOURS_SECONDS
            )

            scenario["breakout_timestamp"] = breakout_close_time

            log(
                "SHORT breakdown confirmed. "
                f"4H close: {candle['close']:.2f}"
            )

            return breakout_close_time

    return None


def find_invalidation(direction, plan, scenario, candles_4h):
    stop_loss = float(plan["stop_loss"])

    if direction == "LONG":
        reference_timestamp = scenario.get("sweep_timestamp")
    elif direction == "SHORT":
        reference_timestamp = scenario.get("breakout_timestamp")
    else:
        raise ValueError(f"Invalid direction: {direction}")

    if reference_timestamp is None:
        return None

    for candle in candles_4h:
        candle_close_time = (
            candle["timestamp"] + FOUR_HOURS_SECONDS
        )

        if direction == "LONG":
            if candle["timestamp"] < reference_timestamp:
                continue

            if candle["close"] < stop_loss:
                return candle

        else:
            if candle_close_time < reference_timestamp:
                continue

            if candle["close"] > stop_loss:
                return candle

    return None


def find_retest(direction, plan, breakout_timestamp, candles_5m):
    zone_low = float(plan["entry_zone_low"])
    zone_high = float(plan["entry_zone_high"])

    for candle in candles_5m:
        candle_close_time = (
            candle["timestamp"] + FIVE_MINUTES_SECONDS
        )

        if candle_close_time <= breakout_timestamp:
            continue

        overlaps_zone = (
            candle["low"] <= zone_high
            and candle["high"] >= zone_low
        )

        if not overlaps_zone:
            continue

        if direction == "LONG":
            confirmed = (
                candle["close"] > candle["open"]
                and candle["close"] > zone_high
            )

            if confirmed:
                return candle

        elif direction == "SHORT":
            confirmed = (
                candle["close"] < candle["open"]
                and candle["close"] < zone_low
            )

            if confirmed:
                return candle

        else:
            raise ValueError(f"Invalid direction: {direction}")

    return None


def build_alert(direction, plan, candle):
    entry_low = float(plan["entry_zone_low"])
    entry_high = float(plan["entry_zone_high"])

    confirmation_time = datetime.fromtimestamp(
        candle["timestamp"] + FIVE_MINUTES_SECONDS,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"BTC {direction} SIGNAL CONFIRMED\n\n"
        f"Symbol: BTC-USDT\n"
        f"Confirmation: {confirmation_time}\n"
        f"Timeframes: 4H + 5M\n\n"
        f"Entry Zone: {entry_low:.2f} - {entry_high:.2f}\n"
        f"Stop Loss: {float(plan['stop_loss']):.2f}\n"
        f"Take Profit 1: {float(plan['take_profit_1']):.2f}\n"
        f"Take Profit 2: {float(plan['take_profit_2']):.2f}\n"
        f"Take Profit 3: {float(plan['take_profit_3']):.2f}\n\n"
        f"5M Candle Close: {candle['close']:.2f}\n"
        f"Source: OKX\n\n"
        "Manual execution only. No automatic trade was opened."
    )


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        log("Telegram credentials are missing.")
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
        raise RuntimeError("Telegram rejected the message.")

    return True


def process_scenario(
    direction,
    plan,
    state,
    candles_4h,
    new_candles_5m,
):
    validate_plan(direction, plan)

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

    if direction == "LONG":
        breakout_timestamp = find_long_activation(
            plan,
            scenario,
            candles_4h,
        )
    else:
        breakout_timestamp = find_short_breakdown(
            plan,
            scenario,
            candles_4h,
        )

    if breakout_timestamp == "INVALIDATED":
        scenario["invalidated"] = True
        scenario["pending_retest"] = None
        log("LONG: scenario invalidated during activation.")
        return

    if breakout_timestamp is None:
        if direction == "LONG":
            if scenario.get("sweep_timestamp") is None:
                log("LONG: waiting for a liquidity sweep.")
            else:
                log("LONG: waiting for a confirmed 4H reclaim.")
        else:
            log("SHORT: waiting for a confirmed 4H breakdown.")
        return

    invalidation = find_invalidation(
        direction,
        plan,
        scenario,
        candles_4h,
    )

    if invalidation is not None:
        scenario["invalidated"] = True
        scenario["pending_retest"] = None

        log(
            f"{direction}: scenario invalidated by 4H candle. "
            f"Close: {invalidation['close']:.2f}"
        )
        return

    retest = scenario.get("pending_retest")

    if retest is None:
        retest = find_retest(
            direction,
            plan,
            breakout_timestamp,
            new_candles_5m,
        )

    if retest is None:
        log(f"{direction}: waiting for a confirmed 5M retest.")
        return

    scenario["pending_retest"] = retest

    message = build_alert(direction, plan, retest)

    log(f"{direction}: retest confirmed.")
    log(message)

    try:
        sent = send_telegram(message)
    except Exception as error:
        log(f"{direction}: Telegram delivery error: {error}")
        sent = False

    if sent:
        scenario["alert_sent"] = True
        scenario["pending_retest"] = None

        fingerprint = scenario["fingerprint"]

        if fingerprint not in state["sent_fingerprints"]:
            state["sent_fingerprints"].append(fingerprint)

        save_json(STATE_FILE, state)

        log(f"{direction}: Telegram alert sent successfully.")

    else:
        log(
            f"{direction}: alert remains pending. "
            "It will be retried on the next run."
        )


def main():
    log("Starting BTC signal monitor.")

    signal = load_json(SIGNAL_FILE)
    state = load_state()

    candles_4h = fetch_candles(
        "4H",
        CANDLE_LIMIT_4H,
    )

    candles_5m = fetch_candles(
        "5m",
        CANDLE_LIMIT_5M,
    )

    log(
        f"Loaded {len(candles_4h)} confirmed 4H candles "
        f"and {len(candles_5m)} confirmed 5M candles."
    )

    latest_5m_timestamp = candles_5m[-1]["timestamp"]
    last_processed = state["last_processed_5m_timestamp"]

    if last_processed is None:
        new_candles_5m = []

        log(
            "Initial run: establishing 5M baseline. "
            "No historical retest will trigger an alert."
        )

    else:
        recovered_candles = fetch_candles_since(
            "5m",
            candles_5m,
            last_processed,
            FIVE_MINUTES_SECONDS,
        )

        new_candles_5m = [
            candle
            for candle in recovered_candles
            if candle["timestamp"] > last_processed
        ]

    log(f"New confirmed 5M candles: {len(new_candles_5m)}")

    processing_successful = True

    for direction, settings in DIRECTIONS.items():
        plan = signal.get(settings["plan_key"])

        if not isinstance(plan, dict):
            processing_successful = False

            log(
                f"{direction}: plan is missing or invalid. "
                "Checkpoint will not advance."
            )
            continue

        try:
            process_scenario(
                direction,
                plan,
                state,
                candles_4h,
                new_candles_5m,
            )

        except Exception as error:
            processing_successful = False

            log(
                f"{direction}: scenario processing error: {error}"
            )

    if processing_successful:
        state["last_processed_5m_timestamp"] = latest_5m_timestamp
        log("All scenarios processed successfully.")

    else:
        log(
            "WARNING: Scenario processing failed. "
            "The 5M checkpoint will not advance."
        )

    save_json(STATE_FILE, state)

    log("BTC signal monitoring completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"MONITOR ERROR: {error}")
        sys.exit(1)
