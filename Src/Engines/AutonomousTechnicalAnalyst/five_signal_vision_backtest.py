
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from .config import Config
from .market_features import build_market_features
from .vision_prompt import build_btc_4h_prompt
from .vision_provider import ask_openrouter_vision


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT_DIR / "Data" / "Archive" / "BTC_USDT_SPOT_4H.csv"
REPORT_DIR = ROOT_DIR / "Reports" / "Five_Signal_Vision_Backtest"

MODEL_ID = os.getenv(
    "OPENROUTER_VISION_MODEL",
    "nvidia/nemotron-3.5-super-vl:free",
).strip()

SAMPLE_COUNT = 5
CHART_CANDLES = 120
MIN_FEATURE_CANDLES = 220
FUTURE_CANDLES = 24

FEE_RATE_PER_SIDE = 0.0005
SLIPPAGE_RATE_PER_SIDE = 0.0002


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
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def load_data():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Market data file not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=required)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    df = df.reset_index(drop=True)

    if len(df) < MIN_FEATURE_CANDLES + FUTURE_CANDLES + SAMPLE_COUNT:
        raise ValueError(
            f"Not enough candles. Found {len(df)}; "
            f"at least {MIN_FEATURE_CANDLES + FUTURE_CANDLES + SAMPLE_COUNT} required."
        )

    if (df["close"] <= 0).any():
        raise ValueError("Invalid non-positive close price.")

    if (
        (df["high"] < df["low"]).any()
        or (df["high"] < df["open"]).any()
        or (df["high"] < df["close"]).any()
        or (df["low"] > df["open"]).any()
        or (df["low"] > df["close"]).any()
    ):
        raise ValueError("Invalid OHLC candle values.")

    return df


def select_historical_indices(df):
    first_index = MIN_FEATURE_CANDLES - 1
    last_index = len(df) - FUTURE_CANDLES - 1

    if last_index <= first_index:
        raise ValueError(
            "Insufficient historical range for five samples."
        )

    indices = np.linspace(
        first_index,
        last_index,
        SAMPLE_COUNT,
        dtype=int,
    )

    indices = sorted(set(indices.tolist()))

    if len(indices) != SAMPLE_COUNT:
        raise ValueError(
            "Could not select five distinct historical cutoffs."
        )

    return indices


def build_context(historical_df, features):
    candle = historical_df.iloc[-1]
    feature_row = features.iloc[-1]

    feature_data = {
        str(key): json_safe(value)
        for key, value in feature_row.to_dict().items()
    }

    timestamp = int(candle["timestamp"])

    return {
        "symbol": "BTC-USDT",
        "timeframe": "4H",
        "data_source": "OKX",
        "analysis_type": "Historical point-in-time analysis",
        "lookahead_used_for_analysis": False,
        "candle_timestamp_ms": timestamp,
        "candle_time_utc": pd.to_datetime(
            timestamp,
            unit="ms",
            utc=True,
        ).isoformat(),
        "historical_candles_available": len(historical_df),
        "current_price": float(candle["close"]),
        "latest_candle": {
            key: float(candle[key])
            for key in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        },
        "calculated_market_features": feature_data,
    }


def create_chart(df, output_path, title):
    chart_df = df.tail(CHART_CANDLES).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    for index, candle in chart_df.iterrows():
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])

        rising = close_price >= open_price
        candle_color = "#16a34a" if rising else "#dc2626"

        ax.plot(
            [index, index],
            [low_price, high_price],
            color=candle_color,
            linewidth=1.0,
        )

        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)

        if body_height == 0:
            body_height = max(abs(close_price) * 0.0001, 0.01)

        rectangle = patches.Rectangle(
            (index - 0.3, body_bottom),
            0.6,
            body_height,
            facecolor=candle_color,
            edgecolor=candle_color,
            linewidth=0.8,
        )

        ax.add_patch(rectangle)

    ax.set_title(title)
    ax.set_xlabel("Historical 4H candles")
    ax.set_ylabel("BTC-USDT Price")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def extract_json(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty AI response.")

    cleaned = content.strip()

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

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(
            "No complete JSON object found in AI response."
        )

    result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("AI response JSON is not an object.")

    return result


def validate_analysis(analysis):
    if analysis.get("decision") not in (
        "LONG",
        "SHORT",
        "WAIT",
    ):
        raise ValueError("Invalid decision value.")

    if analysis.get("market_bias") not in (
        "BULLISH",
        "BEARISH",
        "SIDEWAYS",
    ):
        raise ValueError("Invalid market_bias value.")

    if analysis.get("trade_setup", {}).get("direction") not in (
        "LONG",
        "SHORT",
        "NONE",
    ):
        raise ValueError("Invalid trade_setup direction.")

    return analysis


def get_number(data, key):
    value = data.get(key)

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def evaluate_trade(analysis, future_df):
    decision = analysis.get("decision")
    setup = analysis.get("trade_setup", {})
    direction = setup.get("direction")

    result = {
        "decision": decision,
        "direction": direction,
        "status": "NO_TRADE",
        "entry_price": None,
        "exit_price": None,
        "exit_reason": None,
        "gross_r": None,
        "estimated_cost_r": None,
        "net_r": None,
        "bars_to_entry": None,
        "bars_in_trade": None,
    }

    if decision == "WAIT":
        result["status"] = "WAIT"
        return result

    if decision not in ("LONG", "SHORT"):
        result["status"] = "INVALID_DECISION"
        return result

    if direction != decision:
        result["status"] = "DIRECTION_MISMATCH"
        return result

    zone_low = get_number(setup, "entry_zone_low")
    zone_high = get_number(setup, "entry_zone_high")
    stop_loss = get_number(setup, "stop_loss")
    take_profit = get_number(setup, "take_profit_1")

    if None in (
        zone_low,
        zone_high,
        stop_loss,
        take_profit,
    ):
        result["status"] = "INCOMPLETE_TRADE_SETUP"
        return result

    zone_low, zone_high = sorted(
        [zone_low, zone_high]
    )

    if zone_low <= 0 or zone_high <= 0:
        result["status"] = "INVALID_ENTRY_ZONE"
        return result

    if decision == "LONG":
        if not (
            stop_loss < zone_low
            and take_profit > zone_high
        ):
            result["status"] = "INVALID_LONG_LEVELS"
            return result
    else:
        if not (
            stop_loss > zone_high
            and take_profit < zone_low
        ):
            result["status"] = "INVALID_SHORT_LEVELS"
            return result

    entry_index = None
    entry_price = None

    for index, candle in future_df.reset_index(drop=True).iterrows():
        candle_low = float(candle["low"])
        candle_high = float(candle["high"])

        if (
            candle_low <= zone_high
            and candle_high >= zone_low
        ):
            entry_index = index
            entry_price = (zone_low + zone_high) / 2.0
            break

    if entry_index is None:
        result["status"] = "ENTRY_NOT_REACHED"
        return result

    risk_per_unit = abs(entry_price - stop_loss)

    if risk_per_unit <= 0:
        result["status"] = "INVALID_RISK"
        return result

    result["entry_price"] = entry_price
    result["bars_to_entry"] = entry_index + 1

    exit_price = None
    exit_reason = None
    exit_index = None

    remaining = future_df.reset_index(drop=True).iloc[
        entry_index:
    ]

    for index, candle in remaining.iterrows():
        candle_low = float(candle["low"])
        candle_high = float(candle["high"])

        if decision == "LONG":
            stop_hit = candle_low <= stop_loss
            target_hit = candle_high >= take_profit
        else:
            stop_hit = candle_high >= stop_loss
            target_hit = candle_low <= take_profit

        if stop_hit:
            exit_price = stop_loss
            exit_reason = (
                "STOP_LOSS"
                if not target_hit
                else "STOP_FIRST_SAME_CANDLE"
            )
            exit_index = entry_index + index
            break

        if target_hit:
            exit_price = take_profit
            exit_reason = "TAKE_PROFIT_1"
            exit_index = entry_index + index
            break

    if exit_price is None:
        last_candle = future_df.reset_index(drop=True).iloc[-1]
        exit_price = float(last_candle["close"])
        exit_reason = "HORIZON_CLOSE"
        exit_index = len(future_df) - 1

    if decision == "LONG":
        gross_r = (
            (exit_price - entry_price)
            / risk_per_unit
        )
    else:
        gross_r = (
            (entry_price - exit_price)
            / risk_per_unit
        )

    cost_rate = (
        FEE_RATE_PER_SIDE
        + SLIPPAGE_RATE_PER_SIDE
    )

    estimated_cost = (
        (entry_price + exit_price)
        * cost_rate
        / risk_per_unit
    )

    net_r = gross_r - estimated_cost

    result.update({
        "status": "CLOSED",
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_r": round(gross_r, 4),
        "estimated_cost_r": round(estimated_cost, 4),
        "net_r": round(net_r, 4),
        "bars_in_trade": exit_index - entry_index + 1,
    })

    return result


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not Config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    print("=" * 70)
    print("FIVE-SIGNAL BTC VISION BACKTEST")
    print("=" * 70)

    df = load_data()

    print(f"Loaded candles: {len(df)}")
    print(
        "First candle:",
        pd.to_datetime(
            int(df.iloc[0]["timestamp"]),
            unit="ms",
            utc=True,
        ),
    )
    print(
        "Last candle:",
        pd.to_datetime(
            int(df.iloc[-1]["timestamp"]),
            unit="ms",
            utc=True,
        ),
    )
    print(f"Vision model: {MODEL_ID}")

    indices = select_historical_indices(df)

    results = []

    for sample_number, cutoff_index in enumerate(
        indices,
        start=1,
    ):
        print()
        print("-" * 70)
        print(
            f"SAMPLE {sample_number}/{SAMPLE_COUNT} "
            f"| CUTOFF INDEX: {cutoff_index}"
        )

        historical_df = df.iloc[
            :cutoff_index + 1
        ].copy()

        future_df = df.iloc[
            cutoff_index + 1:
            cutoff_index + 1 + FUTURE_CANDLES
        ].copy()

        if len(future_df) < FUTURE_CANDLES:
            raise ValueError(
                f"Sample {sample_number} has insufficient future candles."
            )

        cutoff_timestamp = int(
            historical_df.iloc[-1]["timestamp"]
        )

        cutoff_time = pd.to_datetime(
            cutoff_timestamp,
            unit="ms",
            utc=True,
        )

        sample_dir = REPORT_DIR / f"sample_{sample_number}"
        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        features = build_market_features(
            historical_df.copy()
        )

        if features.empty:
            raise ValueError(
                f"Feature calculation failed for sample {sample_number}."
            )

        context = build_context(
            historical_df,
            features,
        )

        context_path = sample_dir / "market_context.json"
        save_json(context_path, context)

        chart_path = sample_dir / "vision_chart.png"

        create_chart(
            historical_df,
            chart_path,
            f"BTC-USDT 4H | Historical cutoff: {cutoff_time}",
        )

        prompt = build_btc_4h_prompt(context)

        print(f"Cutoff time: {cutoff_time}")
        print(f"Historical candles: {len(historical_df)}")
        print("Sending chart and context to Vision model...")

        provider_result = ask_openrouter_vision(
            image_path=chart_path,
            prompt=prompt,
            model_override=MODEL_ID,
        )

        if provider_result.get("status") != "success":
            error_result = {
                "sample": sample_number,
                "cutoff_time_utc": cutoff_time.isoformat(),
                "status": "VISION_API_ERROR",
                "model": provider_result.get("model"),
                "error": provider_result.get("error"),
            }

            save_json(
                sample_dir / "error.json",
                error_result,
            )

            results.append(error_result)

            print(
                "Vision API error:",
                provider_result.get("error"),
            )

            continue

        raw_content = provider_result.get("content", "")

        with open(
            sample_dir / "vision_raw.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(raw_content)

        try:
            analysis = extract_json(raw_content)
            validate_analysis(analysis)

        except Exception as error:
            error_result = {
                "sample": sample_number,
                "cutoff_time_utc": cutoff_time.isoformat(),
                "status": "INVALID_AI_OUTPUT",
                "model": provider_result.get("model"),
                "error": str(error),
                "raw_file": "vision_raw.txt",
            }

            save_json(
                sample_dir / "error.json",
                error_result,
            )

            results.append(error_result)

            print("Invalid AI output:", error)
            continue

        save_json(
            sample_dir / "vision_analysis.json",
            analysis,
        )

        evaluation = evaluate_trade(
            analysis,
            future_df,
        )

        sample_result = {
            "sample": sample_number,
            "cutoff_time_utc": cutoff_time.isoformat(),
            "cutoff_price": float(
                historical_df.iloc[-1]["close"]
            ),
            "model": provider_result.get("model"),
            "status": "SUCCESS",
            "market_bias": analysis.get("market_bias"),
            "confidence": analysis.get("confidence"),
            "decision": analysis.get("decision"),
            "evaluation": evaluation,
        }

        save_json(
            sample_dir / "evaluation.json",
            sample_result,
        )

        results.append(sample_result)

        print("Market bias:", analysis.get("market_bias"))
        print("Decision:", analysis.get("decision"))
        print("Confidence:", analysis.get("confidence"))
        print("Evaluation:", evaluation["status"])

        if evaluation.get("net_r") is not None:
            print("Net R:", evaluation["net_r"])

    successful = [
        item for item in results
        if item.get("status") == "SUCCESS"
    ]

    closed_trades = [
        item["evaluation"]
        for item in successful
        if item["evaluation"].get("status") == "CLOSED"
    ]

    net_results = [
        item["net_r"]
        for item in closed_trades
        if item.get("net_r") is not None
    ]

    wins = sum(value > 0 for value in net_results)
    losses = sum(value < 0 for value in net_results)

    gross_profit = sum(
        value for value in net_results
        if value > 0
    )

    gross_loss = abs(sum(
        value for value in net_results
        if value < 0
    ))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else None
    )

    summary = {
        "experiment": "Five historical BTC 4H Vision signals",
        "symbol": "BTC-USDT",
        "timeframe": "4H",
        "model": MODEL_ID,
        "sample_count": SAMPLE_COUNT,
        "successful_vision_analyses": len(successful),
        "api_or_output_errors": SAMPLE_COUNT - len(successful),
        "wait_count": sum(
            item.get("decision") == "WAIT"
            for item in successful
        ),
        "long_count": sum(
            item.get("decision") == "LONG"
            for item in successful
        ),
        "short_count": sum(
            item.get("decision") == "SHORT"
            for item in successful
        ),
        "closed_trade_count": len(closed_trades),
        "wins": wins,
        "losses": losses,
        "win_rate_percent": (
            round(wins / len(closed_trades) * 100, 2)
            if closed_trades
            else None
        ),
        "total_net_r": round(sum(net_results), 4),
        "profit_factor": (
            round(profit_factor, 4)
            if profit_factor is not None
            else None
        ),
        "fee_rate_per_side": FEE_RATE_PER_SIDE,
        "slippage_rate_per_side": SLIPPAGE_RATE_PER_SIDE,
        "future_evaluation_candles": FUTURE_CANDLES,
        "results": results,
        "limitations": [
            "Only five historical analysis points are tested.",
            "The model receives historical data available at each cutoff only.",
            "Only trade_setup LONG or SHORT outputs with complete numeric levels are simulated.",
            "WAIT decisions are not converted into trades.",
            "Entry is simulated at the midpoint of the entry zone when a future candle touches that zone.",
            "If stop loss and target are both touched in one candle, stop loss is assumed first.",
            "Only take_profit_1 is used as the exit target.",
            "Fees and slippage are estimated, not actual exchange fills.",
            "Five samples are not sufficient to establish statistical profitability.",
        ],
    }

    save_json(
        REPORT_DIR / "summary.json",
        summary,
    )

    flat_rows = []

    for item in results:
        evaluation = item.get("evaluation", {})

        flat_rows.append({
            "sample": item.get("sample"),
            "cutoff_time_utc": item.get("cutoff_time_utc"),
            "market_bias": item.get("market_bias"),
            "decision": item.get("decision"),
            "confidence": item.get("confidence"),
            "evaluation_status": evaluation.get("status"),
            "direction": evaluation.get("direction"),
            "entry_price": evaluation.get("entry_price"),
            "exit_price": evaluation.get("exit_price"),
            "exit_reason": evaluation.get("exit_reason"),
            "gross_r": evaluation.get("gross_r"),
            "estimated_cost_r": evaluation.get("estimated_cost_r"),
            "net_r": evaluation.get("net_r"),
        })

    pd.DataFrame(flat_rows).to_csv(
        REPORT_DIR / "summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FIVE-SIGNAL BACKTEST COMPLETED")
    print("=" * 70)
    print("Successful analyses:", len(successful))
    print("Closed trades:", len(closed_trades))
    print("WAIT decisions:", summary["wait_count"])
    print("LONG decisions:", summary["long_count"])
    print("SHORT decisions:", summary["short_count"])
    print("Win rate:", summary["win_rate_percent"])
    print("Total net R:", summary["total_net_r"])
    print("Profit factor:", summary["profit_factor"])
    print("Reports:", REPORT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"FATAL ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
