
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

MAX_AI_CALLS = 40
SIGNAL_INTERVAL = 12
ENTRY_VALIDITY_CANDLES = 3
MAX_HOLDING_CANDLES = 24
WARMUP_CANDLES = 200

FEE_RATE_PER_SIDE = 0.0005
SLIPPAGE_RATE_PER_SIDE = 0.0002
RISK_PER_TRADE = 0.01

INITIAL_EQUITY = 10000.0

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

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (np.bool_,)):
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

    current_timestamp = str(
        current_candle["timestamp_utc"]
    )

    current_close = safe_float(current_candle["close"])

    prompt_data = {
        "symbol": Config.SYMBOL,
        "market_type": Config.MARKET_TYPE,
        "timeframe": Config.TIMEFRAME,
        "decision_time_utc": current_timestamp,
        "current_close": current_close,
        "recent_market_history": snapshot,
    }

    return f"""
You are an independent cryptocurrency technical analyst.

Analyze the following historical BTC-USDT market data.

IMPORTANT RULES:
1. Use only the information supplied in this prompt.
2. Do not assume knowledge of future candles.
3. Do not invent market prices or indicators.
4. Evaluate trend, momentum, volatility, market structure,
   support, resistance, and volume.
5. Avoid entering trades when the evidence is insufficient.
6. Prefer WAIT when the setup is unclear.
7. Do not force a trade.
8. Return exactly one JSON object.
9. Do not include Markdown or explanations outside the JSON.
10. Keep the reason and invalidation fields short.
11. Do not provide hidden reasoning or lengthy analysis.

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

    actual_rr = actual_reward / actual_risk

    if actual_rr <= 0:
        raise ValueError("Invalid risk/reward ratio.")

    result["risk_reward"] = actual_rr

    return result


def get_ai_decision(prompt, cache):
    cache_key = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    if cache_key in cache:
        cached = cache[cache_key]

        if cached.get("status") == "success":
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
    entry = decision["entry"]

    last_index = min(
        decision_index + ENTRY_VALIDITY_CANDLES,
        len(data) - 1,
    )

    for index in range(decision_index + 1, last_index + 1):
        candle = data.iloc[index]

        low = safe_float(candle["low"])
        high = safe_float(candle["high"])

        if low is None or high is None:
            continue

        if low <= entry <= high:
            return index, entry

    return None, None


def execute_trade(data, decision, entry_index, entry_price):
    direction = decision["decision"]

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
    else:
        if not target < entry_price < stop:
            return None

    execution_cost = (
        FEE_RATE_PER_SIDE + SLIPPAGE_RATE_PER_SIDE
    )

    if direction == "LONG":
        executed_entry = entry_price * (1 + execution_cost)

        if not stop < executed_entry < target:
            return None

    else:
        executed_entry = entry_price * (1 - execution_cost)

        if not target < executed_entry < stop:
            return None

    initial_risk = abs(executed_entry - stop)

    if not np.isfinite(initial_risk) or initial_risk <= 0:
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

        if high is None or low is None or high < low:
            return None

        if direction == "LONG":
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target

        if index == entry_index:
            # OHLC data cannot establish intrabar event order.
            # A stop touch on the entry candle is treated conservatively.
            # Target-only touches on the entry candle are ignored.

            if stop_hit:
                exit_index = index
                exit_price = stop
                exit_reason = "STOP_LOSS_ENTRY_CANDLE"
                break

            continue

        # If both levels are touched in one candle,
        # assume the stop-loss was reached first.

        if stop_hit:
            exit_index = index
            exit_price = stop
            exit_reason = "STOP_LOSS"
            break

        if target_hit:
            exit_index = index
            exit_price = target
            exit_reason = "TAKE_PROFIT"
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
        executed_exit = exit_price * (1 - execution_cost)

        gross_pnl = exit_price - entry_price
        net_pnl = executed_exit - executed_entry

    else:
        executed_exit = exit_price * (1 + execution_cost)

        gross_pnl = entry_price - exit_price
        net_pnl = executed_entry - executed_exit

    net_r = net_pnl / initial_risk

    if not np.isfinite(net_r):
        return None

    return {
        "direction": direction,
        "decision_index": decision.get("decision_index"),
        "entry_index": entry_index,
        "exit_index": exit_index,
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
        "initial_risk": initial_risk,
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

    winners = [
        value for value in results
        if value > 0
    ]

    losers = [
        value for value in results
        if value < 0
    ]

    breakeven = [
        value for value in results
        if value == 0
    ]

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

        max_drawdown = max(
            max_drawdown,
            drawdown,
        )

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

    config_status = Config.validate()

    if not config_status["valid"]:
        raise RuntimeError(
            "Configuration error: "
            + "; ".join(config_status["errors"])
        )

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

    if len(data) <= WARMUP_CANDLES:
        raise ValueError("Not enough historical candles.")

    data = data.reset_index(drop=True)

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

    while index < len(data) - 2:
        if index < next_query_index:
            index += 1
            continue

        if api_calls >= MAX_AI_CALLS:
            print("Maximum AI request limit reached.")
            break

        current_time = str(
            data.iloc[index]["timestamp_utc"]
        )

        print(
            f"\nAnalyzing candle {index + 1}/{len(data)} "
            f"| {current_time}"
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

            print(
                "API ERROR:",
                result.get("error"),
            )

            next_query_index = index + SIGNAL_INTERVAL
            index += 1
            continue

        decision = result["decision"]

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
        "symbol": Config.SYMBOL,
        "market_type": Config.MARKET_TYPE,
        "timeframe": Config.TIMEFRAME,
        "total_candles": len(data),
        "first_candle": str(
            data.iloc[0]["timestamp_utc"]
        ),
        "last_candle": str(
            data.iloc[-1]["timestamp_utc"]
        ),
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
        "risk_per_trade": RISK_PER_TRADE,
        "initial_equity": INITIAL_EQUITY,
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
        "execution_status",
        "entry_index",
        "exit_index",
        "error",
        "model",
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

    print(
        f"Win rate: {overall['win_rate_percent']:.2f}%"
    )

    print(f"Net R: {overall['net_r']:.4f}")

    print(
        f"Profit factor: {overall['profit_factor']:.4f}"
        if np.isfinite(overall["profit_factor"])
        else "Profit factor: Infinity"
    )

    print(f"Average R: {overall['average_r']:.4f}")

    print(
        f"Maximum drawdown: "
        f"{overall['max_drawdown_percent']:.2f}%"
    )

    print(
        f"Final equity: "
        f"{overall['final_equity']:.2f}"
    )

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
