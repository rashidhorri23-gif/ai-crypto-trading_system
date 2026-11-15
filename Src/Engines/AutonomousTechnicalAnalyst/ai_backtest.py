
import os
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .ai_providers import ask_openrouter
from .market_features import prepare_market_features


MODEL_ID = "baidu/ernie-5.5-flash-preview:free"
PROMPT_VERSION = "2.0"

MAX_AI_CALLS = 40
SIGNAL_INTERVAL = 12
ENTRY_VALIDITY_CANDLES = 3
MAX_HOLDING_CANDLES = 24
WARMUP_CANDLES = 200

FEE_RATE_PER_SIDE = 0.0005
SLIPPAGE_RATE_PER_SIDE = 0.0002
EXECUTION_COST_PER_SIDE = (
    FEE_RATE_PER_SIDE + SLIPPAGE_RATE_PER_SIDE
)

RISK_PER_TRADE = 0.01
INITIAL_EQUITY = 10000.0

TIMEFRAME_SECONDS = {
    "1M": 60,
    "5M": 300,
    "15M": 900,
    "30M": 1800,
    "1H": 3600,
    "2H": 7200,
    "4H": 14400,
    "6H": 21600,
    "12H": 43200,
    "1D": 86400,
}

CACHE_DIRECTORY = Path("Data/Backtest")
CACHE_FILE = CACHE_DIRECTORY / "ernie_response_cache.json"
TRADES_FILE = CACHE_DIRECTORY / "ernie_backtest_trades.csv"
SUMMARY_FILE = CACHE_DIRECTORY / "ernie_backtest_summary.json"
DECISIONS_FILE = CACHE_DIRECTORY / "ernie_backtest_decisions.csv"


def safe_float(value, default=None):
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def load_cache():
    if not CACHE_FILE.exists():
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        print("WARNING: Cache file could not be read.")

    return {}


def save_cache(cache):
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    temporary_file = CACHE_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(cache),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary_file.replace(CACHE_FILE)


def get_timeframe_seconds():
    timeframe = str(Config.TIMEFRAME).strip().upper()

    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}"
        )

    return TIMEFRAME_SECONDS[timeframe]


def validate_backtest_data(data):
    if data.empty:
        raise ValueError("Market data is empty.")

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp_utc",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing backtest columns: {missing}"
        )

    timestamps = pd.to_numeric(
        data["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Invalid timestamps found.")

    timestamp_values = timestamps.to_numpy(dtype=np.float64)

    if not np.isfinite(timestamp_values).all():
        raise ValueError("Non-finite timestamps found.")

    if (timestamp_values <= 0).any():
        raise ValueError("Non-positive timestamps found.")

    if not np.equal(
        timestamp_values,
        np.floor(timestamp_values),
    ).all():
        raise ValueError("Timestamps must be integer milliseconds.")

    if pd.Series(timestamp_values).duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    expected_interval_ms = get_timeframe_seconds() * 1000

    differences = np.diff(timestamp_values)

    if len(differences) and not np.all(
        differences == expected_interval_ms
    ):
        raise ValueError(
            "Market data contains missing or irregular candle intervals."
        )

    price_columns = ["open", "high", "low", "close"]

    prices = data[price_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if prices.isna().any().any():
        raise ValueError("Invalid OHLC prices found.")

    if not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite OHLC prices found.")

    if (prices <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")

    invalid_candles = (
        (prices["high"] < prices["low"])
        | (prices["high"] < prices["open"])
        | (prices["high"] < prices["close"])
        | (prices["low"] > prices["open"])
        | (prices["low"] > prices["close"])
    )

    if invalid_candles.any():
        raise ValueError("Invalid OHLC candle geometry found.")

    return True


def build_market_snapshot(data, index, history_size=20):
    start_index = max(0, index - history_size + 1)

    selected_columns = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "atr_14",
        "adx_14",
        "plus_di",
        "minus_di",
        "relative_volume",
        "last_swing_high",
        "last_swing_low",
        "bos_bullish",
        "bos_bearish",
        "market_structure",
        "support",
        "resistance",
        "ema_trend",
    ]

    available_columns = [
        column
        for column in selected_columns
        if column in data.columns
    ]

    records = data.iloc[start_index:index + 1][
        available_columns
    ].to_dict(orient="records")

    return json_safe(records)


def build_prompt(data, index):
    current_candle = data.iloc[index]

    snapshot = build_market_snapshot(data, index)

    candle_open_time = pd.Timestamp(
        current_candle["timestamp_utc"]
    )

    candle_close_time = (
        candle_open_time
        + pd.Timedelta(seconds=get_timeframe_seconds())
    )

    current_close = safe_float(current_candle["close"])

    prompt_data = {
        "symbol": Config.SYMBOL,
        "market_type": Config.MARKET_TYPE,
        "timeframe": Config.TIMEFRAME,
        "candle_open_time_utc": candle_open_time.isoformat(),
        "decision_time_utc": candle_close_time.isoformat(),
        "current_close": current_close,
        "recent_market_history": snapshot,
    }

    return f"""
You are an independent cryptocurrency technical analyst.

Analyze the following historical BTC-USDT market data.

IMPORTANT RULES:
1. Use only the information supplied in this prompt.
2. The current candle is fully closed.
3. Do not assume knowledge of future candles.
4. Do not invent market prices or indicators.
5. Evaluate trend, momentum, volatility, market structure,
   support, resistance, and volume.
6. Avoid entering trades when the evidence is insufficient.
7. Prefer WAIT when the setup is unclear.
8. Do not force a trade.
9. Return exactly one JSON object.
10. Do not include Markdown or explanations outside the JSON.
11. Keep the reason and invalidation fields short.
12. Do not provide hidden reasoning or lengthy analysis.

A LONG decision requires:
- entry above stop_loss
- take_profit above entry
- positive risk_reward

A SHORT decision requires:
- stop_loss above entry
- take_profit below entry
- positive risk_reward

A WAIT decision must use zero for:
entry, stop_loss, take_profit, risk_reward.

Return this exact JSON structure:

{{
  "decision": "LONG or SHORT or WAIT",
  "bias": "BULLISH or BEARISH or NEUTRAL",
  "confidence": 0,
  "entry": 0,
  "stop_loss": 0,
  "take_profit": 0,
  "risk_reward": 0,
  "reason": "Short technical explanation",
  "invalidation": "Condition that invalidates the setup"
}}

Historical market data:

{json.dumps(prompt_data, ensure_ascii=False)}
""".strip()


def extract_json(response_text):
    if not isinstance(response_text, str):
        raise ValueError("Model response is not text.")

    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return json.loads(cleaned[start:end + 1])


def validate_decision(payload):
    if not isinstance(payload, dict):
        raise ValueError("Decision must be a JSON object.")

    decision = str(
        payload.get("decision", "")
    ).strip().upper()

    if decision not in ("LONG", "SHORT", "WAIT"):
        raise ValueError("Invalid decision value.")

    bias = str(
        payload.get("bias", "NEUTRAL")
    ).strip().upper()

    if bias not in ("BULLISH", "BEARISH", "NEUTRAL"):
        raise ValueError("Invalid bias value.")

    confidence = safe_float(
        payload.get("confidence"),
        None,
    )

    if confidence is None or not 0 <= confidence <= 100:
        raise ValueError("Confidence must be between 0 and 100.")

    result = {
        "decision": decision,
        "bias": bias,
        "confidence": confidence,
        "entry": safe_float(payload.get("entry"), 0),
        "stop_loss": safe_float(payload.get("stop_loss"), 0),
        "take_profit": safe_float(payload.get("take_profit"), 0),
        "risk_reward": safe_float(payload.get("risk_reward"), 0),
        "reason": str(payload.get("reason", "")),
        "invalidation": str(payload.get("invalidation", "")),
    }

    if decision == "WAIT":
        result["entry"] = 0
        result["stop_loss"] = 0
        result["take_profit"] = 0
        result["risk_reward"] = 0
        return result

    entry = result["entry"]
    stop = result["stop_loss"]
    target = result["take_profit"]

    if not all(
        value > 0
        for value in (entry, stop, target)
    ):
        raise ValueError("Trade prices must be positive.")

    if decision == "LONG":
        if not stop < entry < target:
            raise ValueError("Invalid LONG price geometry.")

    if decision == "SHORT":
        if not target < entry < stop:
            raise ValueError("Invalid SHORT price geometry.")

    actual_risk = abs(entry - stop)
    actual_reward = abs(target - entry)

    if actual_risk <= 0:
        raise ValueError("Invalid trade risk.")

    result["risk_reward"] = actual_reward / actual_risk

    return result


def get_ai_decision(prompt, cache):
    cache_identity = {
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
    }

    cache_key = hashlib.sha256(
        json.dumps(
            cache_identity,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    if cache_key in cache:
        cached = cache[cache_key]

        if (
            isinstance(cached, dict)
            and cached.get("status") == "success"
        ):
            return cached, True

    response = ask_openrouter(
        prompt,
        model_override=MODEL_ID,
    )

    if response.get("status") != "success":
        return {
            "status": "error",
            "error": response.get(
                "error",
                "Unknown API error",
            ),
            "model": response.get("model", MODEL_ID),
            "diagnostics": response.get("diagnostics", {}),
        }, False

    response_text = response.get("response_text", "")

    try:
        parsed = extract_json(response_text)
        decision = validate_decision(parsed)

    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return {
            "status": "error",
            "error": f"Invalid model response: {error}",
            "model": response.get("model", MODEL_ID),
            "diagnostics": response.get("diagnostics", {}),
        }, False

    result = {
        "status": "success",
        "model": response.get("model", MODEL_ID),
        "decision": decision,
        "latency_seconds": response.get("latency_seconds"),
        "usage": response.get("usage"),
        "attempts": response.get("attempts"),
    }

    cache[cache_key] = result
    save_cache(cache)

    return result, False


def find_entry_fill(data, decision_index, decision):
    direction = decision.get("decision")
    entry = safe_float(decision.get("entry"))

    if direction not in ("LONG", "SHORT"):
        return None, None

    if entry is None or entry <= 0:
        return None, None

    if not isinstance(decision_index, (int, np.integer)):
        return None, None

    if decision_index < 0 or decision_index >= len(data):
        return None, None

    last_index = min(
        decision_index + ENTRY_VALIDITY_CANDLES,
        len(data) - 1,
    )

    for index in range(decision_index + 1, last_index + 1):
        candle = data.iloc[index]

        open_price = safe_float(candle["open"])
        low = safe_float(candle["low"])
        high = safe_float(candle["high"])

        if (
            open_price is None
            or low is None
            or high is None
            or high < low
        ):
            continue

        if direction == "LONG":
            if open_price <= entry:
                return index, open_price

            if low <= entry <= high:
                return index, entry

        else:
            if open_price >= entry:
                return index, open_price

            if low <= entry <= high:
                return index, entry

    return None, None


def get_exit_fill(direction, candle, stop, target):
    open_price = safe_float(candle["open"])
    high = safe_float(candle["high"])
    low = safe_float(candle["low"])

    if (
        open_price is None
        or high is None
        or low is None
        or high < low
    ):
        return None, None

    if direction == "LONG":
        if open_price <= stop:
            return open_price, "STOP_LOSS_GAP"

        if open_price >= target:
            return open_price, "TAKE_PROFIT_GAP"

        stop_hit = low <= stop
        target_hit = high >= target

    else:
        if open_price >= stop:
            return open_price, "STOP_LOSS_GAP"

        if open_price <= target:
            return open_price, "TAKE_PROFIT_GAP"

        stop_hit = high >= stop
        target_hit = low <= target

    if stop_hit:
        return stop, "STOP_LOSS"

    if target_hit:
        return target, "TAKE_PROFIT"

    return None, None


def execute_trade(data, decision, entry_index, entry_price):
    direction = decision.get("decision")

    stop = safe_float(decision.get("stop_loss"))
    target = safe_float(decision.get("take_profit"))
    entry_price = safe_float(entry_price)

    if direction not in ("LONG", "SHORT"):
        return None

    if entry_price is None or stop is None or target is None:
        return None

    if entry_price <= 0 or stop <= 0 or target <= 0:
        return None

    if not isinstance(entry_index, (int, np.integer)):
        return None

    if entry_index < 0 or entry_index >= len(data):
        return None

    if direction == "LONG":
        if not stop < entry_price < target:
            return None

        executed_entry = (
            entry_price * (1 + EXECUTION_COST_PER_SIDE)
        )

        planned_stop_exit = (
            stop * (1 - EXECUTION_COST_PER_SIDE)
        )

        planned_risk = executed_entry - planned_stop_exit

    else:
        if not target < entry_price < stop:
            return None

        executed_entry = (
            entry_price * (1 - EXECUTION_COST_PER_SIDE)
        )

        planned_stop_exit = (
            stop * (1 + EXECUTION_COST_PER_SIDE)
        )

        planned_risk = planned_stop_exit - executed_entry

    if (
        not np.isfinite(planned_risk)
        or planned_risk <= 0
    ):
        return None

    last_index = min(
        entry_index + MAX_HOLDING_CANDLES,
        len(data) - 1,
    )

    exit_index = None
    exit_price = None
    exit_reason = None

    for index in range(entry_index, last_index + 1):
        candle = data.iloc[index]

        high = safe_float(candle["high"])
        low = safe_float(candle["low"])
        open_price = safe_float(candle["open"])

        if (
            high is None
            or low is None
            or open_price is None
            or high < low
        ):
            return None

        if index == entry_index:
            # Intrabar ordering is unknown.
            # A stop touch is treated conservatively.
            # Target-only touches are ignored on the entry candle.

            if direction == "LONG":
                stop_hit = low <= stop
            else:
                stop_hit = high >= stop

            if stop_hit:
                exit_index = index

                if direction == "LONG" and open_price <= stop:
                    exit_price = open_price
                    exit_reason = "STOP_LOSS_GAP_ENTRY_CANDLE"

                elif direction == "SHORT" and open_price >= stop:
                    exit_price = open_price
                    exit_reason = "STOP_LOSS_GAP_ENTRY_CANDLE"

                else:
                    exit_price = stop
                    exit_reason = "STOP_LOSS_ENTRY_CANDLE"

                break

            continue

        fill_price, reason = get_exit_fill(
            direction,
            candle,
            stop,
            target,
        )

        if reason is not None:
            exit_index = index
            exit_price = fill_price
            exit_reason = reason
            break

    if exit_index is None:
        exit_index = last_index

        exit_price = safe_float(
            data.iloc[exit_index]["close"]
        )

        if exit_price is None or exit_price <= 0:
            return None

        exit_reason = "TIME_EXIT"

    if direction == "LONG":
        executed_exit = (
            exit_price * (1 - EXECUTION_COST_PER_SIDE)
        )

        gross_pnl = exit_price - entry_price
        net_pnl = executed_exit - executed_entry

    else:
        executed_exit = (
            exit_price * (1 + EXECUTION_COST_PER_SIDE)
        )

        gross_pnl = entry_price - exit_price
        net_pnl = executed_entry - executed_exit

    net_r = net_pnl / planned_risk

    if not np.isfinite(net_r):
        return None

    return {
        "direction": direction,
        "decision_index": decision.get("decision_index"),
        "entry_index": int(entry_index),
        "exit_index": int(exit_index),
        "entry_time": str(
            data.iloc[entry_index]["timestamp_utc"]
        ),
        "exit_time": str(
            data.iloc[exit_index]["timestamp_utc"]
        ),
        "entry_price": entry_price,
        "executed_entry": executed_entry,
        "stop_loss": stop,
        "take_profit": target,
        "exit_price": exit_price,
        "executed_exit": executed_exit,
        "exit_reason": exit_reason,
        "planned_risk_per_unit": planned_risk,
        "initial_risk": planned_risk,
        "gross_pnl_per_unit": gross_pnl,
        "net_pnl_per_unit": net_pnl,
        "net_r": net_r,
        "confidence": decision.get("confidence"),
        "model_reason": decision.get("reason"),
    }


def calculate_performance(trades):
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_percent": 0,
            "net_r": 0,
            "profit_factor": 0,
            "average_r": 0,
            "max_drawdown_percent": 0,
            "initial_equity": INITIAL_EQUITY,
            "final_equity": INITIAL_EQUITY,
        }

    results = [
        safe_float(trade.get("net_r"), 0)
        for trade in trades
    ]

    winners = [value for value in results if value > 0]
    losers = [value for value in results if value < 0]
    breakeven = [value for value in results if value == 0]

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0

    equity = INITIAL_EQUITY
    peak_equity = equity
    max_drawdown = 0.0

    for result in results:
        equity *= 1 + (RISK_PER_TRADE * result)

        peak_equity = max(peak_equity, equity)

        drawdown = (
            (peak_equity - equity) / peak_equity
        ) * 100

        max_drawdown = max(max_drawdown, drawdown)

    return {
        "total_trades": len(results),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "breakeven_trades": len(breakeven),
        "win_rate_percent": (
            len(winners) / len(results)
        ) * 100,
        "net_r": sum(results),
        "profit_factor": profit_factor,
        "average_r": float(np.mean(results)),
        "max_drawdown_percent": max_drawdown,
        "initial_equity": INITIAL_EQUITY,
        "final_equity": equity,
    }


def calculate_direction_performance(trades, direction):
    selected = [
        trade
        for trade in trades
        if trade["direction"] == direction
    ]

    return calculate_performance(selected)


def run_backtest():
    print("=" * 65)
    print("ERNIE AI HISTORICAL BACKTEST")
    print("=" * 65)

    try:
        Config.validate()
    except ValueError as error:
        raise RuntimeError(
            f"Configuration error: {error}"
        ) from error

    data_file = os.path.join(
        Config.DATA_DIRECTORY,
        "BTC_USDT_SPOT_4H.csv",
    )

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"Market data file not found: {data_file}"
        )

    print(f"Data file: {data_file}")
    print(f"Model: {MODEL_ID}")

    data = prepare_market_features(data_file)
    data = data.reset_index(drop=True)

    validate_backtest_data(data)

    if len(data) <= WARMUP_CANDLES:
        raise ValueError("Not enough historical candles.")

    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = load_cache()

    trades = []
    decisions = []

    api_calls = 0
    cache_hits = 0
    api_errors = 0

    index = WARMUP_CANDLES
    next_query_index = WARMUP_CANDLES

    # A decision must have enough future candles for the full
    # entry-validity window and maximum holding period.
    last_decision_index = (
        len(data)
        - 1
        - ENTRY_VALIDITY_CANDLES
        - MAX_HOLDING_CANDLES
    )

    while index <= last_decision_index:
        if index < next_query_index:
            index += 1
            continue

        if api_calls >= MAX_AI_CALLS:
            print("Maximum AI request limit reached.")
            break

        candle_open_time = pd.Timestamp(
            data.iloc[index]["timestamp_utc"]
        )

        decision_time = (
            candle_open_time
            + pd.Timedelta(seconds=get_timeframe_seconds())
        )

        current_time = decision_time.isoformat()

        print(
            f"\nAnalyzing candle {index + 1}/{len(data)} "
            f"| Decision time: {current_time}"
        )

        prompt = build_prompt(data, index)

        result, was_cached = get_ai_decision(
            prompt,
            cache,
        )

        if was_cached:
            cache_hits += 1
        else:
            api_calls += 1

        if result.get("status") != "success":
            api_errors += 1

            decisions.append({
                "timestamp": current_time,
                "decision": "API_ERROR",
                "error": result.get("error"),
                "model": result.get("model"),
            })

            print("API ERROR:", result.get("error"))

            next_query_index = index + SIGNAL_INTERVAL
            index += 1
            continue

        decision = result["decision"].copy()
        decision["decision_index"] = index

        decisions.append({
            "timestamp": current_time,
            "decision": decision["decision"],
            "bias": decision["bias"],
            "confidence": decision["confidence"],
            "entry": decision["entry"],
            "stop_loss": decision["stop_loss"],
            "take_profit": decision["take_profit"],
            "risk_reward": decision["risk_reward"],
            "reason": decision["reason"],
            "invalidation": decision["invalidation"],
            "cached": was_cached,
            "model": result.get("model"),
            "execution_status": "NOT_PROCESSED",
        })

        print(
            f"Decision: {decision['decision']} "
            f"| Confidence: {decision['confidence']}"
        )

        if decision["decision"] == "WAIT":
            decisions[-1]["execution_status"] = "WAIT"

            next_query_index = index + SIGNAL_INTERVAL
            index += 1
            continue

        fill_index, fill_price = find_entry_fill(
            data,
            index,
            decision,
        )

        if fill_index is None:
            print("Entry order was not filled.")

            decisions[-1]["execution_status"] = "NOT_FILLED"

            next_query_index = index + SIGNAL_INTERVAL
            index += 1
            continue

        trade = execute_trade(
            data,
            decision,
            fill_index,
            fill_price,
        )

        if trade is None:
            print("Trade rejected: invalid risk or price data.")

            decisions[-1]["execution_status"] = "REJECTED"

            next_query_index = index + SIGNAL_INTERVAL
            index += 1
            continue

        trades.append(trade)

        decisions[-1]["execution_status"] = "EXECUTED"
        decisions[-1]["entry_index"] = fill_index
        decisions[-1]["exit_index"] = trade["exit_index"]

        print(
            f"Trade closed: {trade['exit_reason']} "
            f"| Net R: {trade['net_r']:.4f}"
        )

        index = trade["exit_index"] + 1

        next_query_index = max(
            next_query_index,
            index,
        )

    overall = calculate_performance(trades)

    long_results = calculate_direction_performance(
        trades,
        "LONG",
    )

    short_results = calculate_direction_performance(
        trades,
        "SHORT",
    )

    summary = {
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "symbol": Config.SYMBOL,
        "market_type": Config.MARKET_TYPE,
        "timeframe": Config.TIMEFRAME,
        "total_candles": len(data),
        "first_candle": str(data.iloc[0]["timestamp_utc"]),
        "last_candle": str(data.iloc[-1]["timestamp_utc"]),
        "warmup_candles": WARMUP_CANDLES,
        "maximum_ai_calls": MAX_AI_CALLS,
        "actual_api_calls": api_calls,
        "cache_hits": cache_hits,
        "api_errors": api_errors,
        "signal_interval": SIGNAL_INTERVAL,
        "entry_validity_candles": ENTRY_VALIDITY_CANDLES,
        "maximum_holding_candles": MAX_HOLDING_CANDLES,
        "fee_rate_per_side": FEE_RATE_PER_SIDE,
        "slippage_rate_per_side": SLIPPAGE_RATE_PER_SIDE,
        "execution_cost_per_side": EXECUTION_COST_PER_SIDE,
        "risk_per_trade": RISK_PER_TRADE,
        "initial_equity": INITIAL_EQUITY,
        "tail_policy": (
            "Only decisions with a complete entry-validity "
            "and maximum-holding window are evaluated."
        ),
        "performance": overall,
        "long_performance": long_results,
        "short_performance": short_results,
        "trades": trades,
    }

    trade_columns = [
        "direction",
        "decision_index",
        "entry_index",
        "exit_index",
        "entry_time",
        "exit_time",
        "entry_price",
        "executed_entry",
        "stop_loss",
        "take_profit",
        "exit_price",
        "executed_exit",
        "exit_reason",
        "planned_risk_per_unit",
        "initial_risk",
        "gross_pnl_per_unit",
        "net_pnl_per_unit",
        "net_r",
        "confidence",
        "model_reason",
    ]

    decision_columns = [
        "timestamp",
        "decision",
        "bias",
        "confidence",
        "entry",
        "stop_loss",
        "take_profit",
        "risk_reward",
        "reason",
        "invalidation",
        "cached",
        "model",
        "execution_status",
        "entry_index",
        "exit_index",
        "error",
    ]

    pd.DataFrame(
        trades,
        columns=trade_columns,
    ).to_csv(
        TRADES_FILE,
        index=False,
    )

    pd.DataFrame(
        decisions,
        columns=decision_columns,
    ).to_csv(
        DECISIONS_FILE,
        index=False,
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(summary),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\n" + "=" * 65)
    print("BACKTEST RESULTS")
    print("=" * 65)

    print(f"Total candles: {len(data)}")
    print(f"AI requests: {api_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"API errors: {api_errors}")

    print(f"Total trades: {overall['total_trades']}")
    print(f"Winning trades: {overall['winning_trades']}")
    print(f"Losing trades: {overall['losing_trades']}")
    print(f"Breakeven trades: {overall['breakeven_trades']}")

    print(f"Win rate: {overall['win_rate_percent']:.2f}%")
    print(f"Net R: {overall['net_r']:.4f}")

    if np.isfinite(overall["profit_factor"]):
        print(f"Profit factor: {overall['profit_factor']:.4f}")
    else:
        print("Profit factor: Infinity")

    print(f"Average R: {overall['average_r']:.4f}")

    print(
        f"Maximum drawdown: "
        f"{overall['max_drawdown_percent']:.2f}%"
    )

    print(f"Final equity: {overall['final_equity']:.2f}")

    print("\nLONG PERFORMANCE")
    print(json.dumps(
        json_safe(long_results),
        indent=2,
        ensure_ascii=False,
    ))

    print("\nSHORT PERFORMANCE")
    print(json.dumps(
        json_safe(short_results),
        indent=2,
        ensure_ascii=False,
    ))

    print("\nOutput files:")
    print(TRADES_FILE)
    print(DECISIONS_FILE)
    print(SUMMARY_FILE)
    print(CACHE_FILE)

    print("\nBACKTEST COMPLETED")

    return summary


if __name__ == "__main__":
    run_backtest()
