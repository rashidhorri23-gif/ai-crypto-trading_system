
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from .market_features import prepare_market_features
from .vision_provider import ask_openrouter_vision
from .vision_prompt import build_btc_analysis_prompt
from .paper_trading import load_ledger


ROOT_DIR = Path(__file__).resolve().parents[3]

ARCHIVE_DIR = ROOT_DIR / "Data" / "Archive"
SIGNAL_DIR = ROOT_DIR / "Data" / "Signals"
OUTPUT_DIR = ROOT_DIR / "Reports"

DATA_FILES = {
    "4H": ARCHIVE_DIR / "BTC_USDT_SPOT_4H.csv",
    "15M": ARCHIVE_DIR / "BTC_USDT_SPOT_15m.csv",
    "5M": ARCHIVE_DIR / "BTC_USDT_SPOT_5m.csv",
}

CURRENT_SIGNAL_FILE = SIGNAL_DIR / "current_btc_signal.json"
STATE_FILE = SIGNAL_DIR / "daily_btc_analysis_state.json"

CHART_FILE = OUTPUT_DIR / "BTC_4H_Vision_Chart.png"
CONTEXT_FILE = OUTPUT_DIR / "BTC_MultiTF_Market_Context.json"
VISION_REPORT_FILE = OUTPUT_DIR / "BTC_4H_Vision_Report.json"
VISION_RAW_FILE = OUTPUT_DIR / "BTC_4H_Vision_Raw.txt"

TIMEFRAME_INTERVALS_MS = {
    "4H": 4 * 60 * 60 * 1000,
    "15M": 15 * 60 * 1000,
    "5M": 5 * 60 * 1000,
}

MAX_DATA_AGE_MINUTES = {
    "4H": 8 * 60,
    "15M": 30,
    "5M": 10,
}

MAX_CONTEXT_CANDLES = {
    "4H": 120,
    "15M": 80,
    "5M": 100,
}

SCHEMA_VERSION = "2.0"


def utc_now():
    return datetime.now(timezone.utc)


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def atomic_write_json(path, data):
    atomic_write_text(
        path,
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
    )


def json_safe(value):
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number.is_integer():
            return int(number)
        return number

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def safe_float(value, field_name, allow_none=False):
    if value is None and allow_none:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")

    return number


def normalize_market_data(df, timeframe):
    required_columns = {
        "timestamp", "open", "high", "low", "close", "volume"
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{timeframe}: missing candle columns: {sorted(missing)}"
        )

    data = df.copy()

    data["timestamp"] = pd.to_numeric(
        data["timestamp"], errors="raise"
    )

    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data["timestamp"].isna().any():
        raise ValueError(f"{timeframe}: null timestamps.")

    if data["timestamp"].duplicated().any():
        raise ValueError(f"{timeframe}: duplicate timestamps.")

    data = data.sort_values("timestamp").reset_index(drop=True)

    if data.empty:
        raise ValueError(f"{timeframe}: empty candle data.")

    numeric_columns = [
        "timestamp", "open", "high", "low", "close", "volume"
    ]

    for column in numeric_columns:
        if not data[column].apply(
            lambda value: math.isfinite(float(value))
        ).all():
            raise ValueError(
                f"{timeframe}: non-finite values in {column}."
            )

    if (data[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{timeframe}: prices must be positive.")

    if (data["volume"] < 0).any():
        raise ValueError(f"{timeframe}: negative volume.")

    invalid = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    if invalid.any():
        raise ValueError(f"{timeframe}: invalid OHLC geometry.")

    return data


def filter_closed_candles(df, timeframe, now):
    interval_ms = TIMEFRAME_INTERVALS_MS[timeframe]
    now_ms = int(now.timestamp() * 1000)
    current_open = (now_ms // interval_ms) * interval_ms

    closed = df[df["timestamp"] < current_open].copy()

    if closed.empty:
        raise ValueError(f"{timeframe}: no closed candles.")

    return closed.reset_index(drop=True)


def get_data_freshness(df, timeframe, now):
    interval_ms = TIMEFRAME_INTERVALS_MS[timeframe]
    latest_timestamp = int(df.iloc[-1]["timestamp"])

    latest_open = pd.to_datetime(
        latest_timestamp, unit="ms", utc=True
    )

    latest_close = latest_open + pd.Timedelta(
        milliseconds=interval_ms
    )

    age_minutes = (
        now - latest_close.to_pydatetime()
    ).total_seconds() / 60

    max_age = MAX_DATA_AGE_MINUTES[timeframe]

    return {
        "timeframe": timeframe,
        "latest_candle_open_utc": latest_open.isoformat(),
        "latest_candle_close_utc": latest_close.isoformat(),
        "age_minutes": round(age_minutes, 2),
        "max_age_minutes": max_age,
        "is_fresh": 0 <= age_minutes <= max_age,
    }


def get_interval_gaps(df, timeframe):
    interval_ms = TIMEFRAME_INTERVALS_MS[timeframe]
    differences = df["timestamp"].diff().dropna()
    gaps = differences[differences > interval_ms]

    return {
        "missing_interval_count": int(len(gaps)),
        "largest_interval_ms": (
            int(gaps.max()) if not gaps.empty else interval_ms
        ),
    }


def load_timeframe_data(timeframe, now):
    file_path = DATA_FILES[timeframe]

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{timeframe} archive not found: {file_path}"
        )

    raw_data = normalize_market_data(
        pd.read_csv(file_path), timeframe
    )

    features = prepare_market_features(file_path)
    features = normalize_market_data(features, timeframe)
    closed = filter_closed_candles(features, timeframe, now)

    if len(closed) < 30:
        raise ValueError(
            f"{timeframe}: insufficient confirmed candle history."
        )

    freshness = get_data_freshness(closed, timeframe, now)
    gaps = get_interval_gaps(closed, timeframe)

    return {
        "data": closed,
        "freshness": freshness,
        "gaps": gaps,
        "raw_candle_count": len(raw_data),
        "closed_candle_count": len(closed),
    }


def draw_candlesticks(ax, df):
    candle_width = 0.65

    for index, row in enumerate(df.itertuples(index=False)):
        bullish = row.close >= row.open
        color = "green" if bullish else "red"

        ax.vlines(
            index, row.low, row.high,
            color=color, linewidth=0.8
        )

        body_bottom = min(row.open, row.close)
        body_height = abs(row.close - row.open)

        if body_height == 0:
            body_height = max(row.close * 0.0001, 0.01)

        ax.add_patch(
            Rectangle(
                (index - candle_width / 2, body_bottom),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
            )
        )


def generate_4h_chart(df):
    chart_data = (
        df.tail(MAX_CONTEXT_CANDLES["4H"])
        .copy()
        .reset_index(drop=True)
    )

    if len(chart_data) < 50:
        raise ValueError("Not enough 4H candles for chart.")

    required = ("ema_20", "ema_50", "ema_200", "timestamp_utc")
    missing = [field for field in required if field not in chart_data.columns]

    if missing:
        raise ValueError(f"Missing chart features: {missing}")

    fig, ax = plt.subplots(figsize=(18, 10))

    try:
        draw_candlesticks(ax, chart_data)

        for column, label in (
            ("ema_20", "EMA 20"),
            ("ema_50", "EMA 50"),
            ("ema_200", "EMA 200"),
        ):
            ax.plot(
                chart_data.index,
                chart_data[column],
                label=label,
                linewidth=1.2,
            )

        latest = chart_data.iloc[-1]

        for column, label in (
            ("support", "Latest Support"),
            ("resistance", "Latest Resistance"),
        ):
            if column in chart_data.columns and pd.notna(latest[column]):
                ax.axhline(
                    latest[column],
                    linestyle="--",
                    linewidth=1,
                    label=label,
                )

        step = max(1, len(chart_data) // 12)
        ticks = list(range(0, len(chart_data), step))

        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                chart_data.iloc[index]["timestamp_utc"].strftime(
                    "%m-%d %H:%M"
                )
                for index in ticks
            ],
            rotation=35,
            ha="right",
        )

        ax.set_title("BTC-USDT | 4H Market Context", fontsize=16)
        ax.set_ylabel("Price (USDT)")
        ax.set_xlabel("UTC Time")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        temporary_chart = CHART_FILE.with_name(
            CHART_FILE.stem + ".tmp.png"
        )

        fig.savefig(
            temporary_chart,
            dpi=180,
            bbox_inches="tight",
        )
        temporary_chart.replace(CHART_FILE)

    finally:
        plt.close(fig)

    return chart_data


def build_timeframe_context(timeframe, result):
    df = result["data"]
    recent = []

    for _, row in df.tail(MAX_CONTEXT_CANDLES[timeframe]).iterrows():
        candle = {
            column: json_safe(row[column])
            for column in df.columns
            if column != "timestamp_utc"
        }

        candle["timestamp_utc"] = pd.to_datetime(
            int(row["timestamp"]), unit="ms", utc=True
        ).isoformat()

        recent.append(candle)

    return {
        "timeframe": timeframe,
        "source": "OKX",
        "freshness": result["freshness"],
        "data_quality": result["gaps"],
        "raw_candle_count": result["raw_candle_count"],
        "closed_candle_count": result["closed_candle_count"],
        "latest_confirmed_candle": recent[-1],
        "recent_confirmed_candles": recent,
        "latest_close": json_safe(df.iloc[-1]["close"]),
    }


def load_previous_signal():
    if not CURRENT_SIGNAL_FILE.is_file():
        return None

    try:
        signal = json.loads(
            CURRENT_SIGNAL_FILE.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"Cannot read previous signal: {error}"
        ) from error

    if not isinstance(signal, dict):
        raise ValueError("Previous signal must be a JSON object.")

    return signal


def load_open_trades():
    ledger = load_ledger()

    if not isinstance(ledger, dict):
        raise ValueError("Paper-trading ledger must be a JSON object.")

    trades = ledger.get("trades", [])

    if not isinstance(trades, list):
        raise ValueError("Ledger trades must be a list.")

    opened = [
        trade for trade in trades
        if isinstance(trade, dict)
        and str(trade.get("status", "")).upper() == "OPEN"
    ]

    if len(opened) == 1:
        return opened[0]

    return opened or None


def build_multi_timeframe_context(results, now):
    contexts = {
        timeframe: build_timeframe_context(
            timeframe, results[timeframe]
        )
        for timeframe in ("4H", "15M", "5M")
    }

    return {
        "symbol": "BTC-USDT",
        "source": "OKX",
        "analysis_trigger": "CLOSED_15M_CANDLE",
        "analysis_time_utc": now.isoformat(),
        "timeframes": contexts,
        "data_status": {
            timeframe: results[timeframe]["freshness"]
            for timeframe in ("4H", "15M", "5M")
        },
        "data_quality": {
            timeframe: results[timeframe]["gaps"]
            for timeframe in ("4H", "15M", "5M")
        },
    }


def load_state():
    if not STATE_FILE.is_file():
        return {}

    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read analysis state: {error}") from error

    if not isinstance(state, dict):
        raise ValueError("Analysis state must be a JSON object.")

    return state


def get_candle_id(timestamp_ms):
    return str(int(timestamp_ms))


def already_analyzed(candle_id):
    state = load_state()
    return state.get("last_successful_15m_candle_id") == candle_id


def save_successful_analysis(candle_id, analysis_id):
    state = load_state()

    state["last_successful_15m_candle_id"] = candle_id
    state["last_successful_analysis_id"] = analysis_id
    state["last_successful_analysis_utc"] = utc_now().isoformat()

    atomic_write_json(STATE_FILE, state)


def validate_timestamp(value, field_name, required=False):
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required.")
        return None

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be an ISO timestamp.")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} is invalid.") from error

    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must use UTC.")

    return parsed


def validate_trade_geometry(direction, entry, stop, targets, label):
    if direction == "LONG":
        valid = (
            stop < entry
            and all(target > entry for target in targets)
            and targets[0] <= targets[1] <= targets[2]
        )
    elif direction == "SHORT":
        valid = (
            stop > entry
            and all(target < entry for target in targets)
            and targets[0] >= targets[1] >= targets[2]
        )
    else:
        raise ValueError(f"{label}: invalid direction.")

    if not valid:
        raise ValueError(f"{label}: invalid trade price geometry.")


def validate_vision_analysis(analysis, previous_signal, open_trade):
    if not isinstance(analysis, dict):
        raise ValueError("AI response must be a JSON object.")

    required = {
        "schema_version", "symbol", "timeframes", "decision",
        "confidence", "market_bias", "entry", "stop_loss",
        "take_profit_1", "take_profit_2", "take_profit_3",
        "reasoning", "main_risks", "data_limitations",
        "previous_signal_action", "open_trade_action",
        "open_trade_updated_levels",
    }

    missing = required.difference(analysis)
    if missing:
        raise ValueError(f"AI response missing fields: {sorted(missing)}")

    if analysis["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported AI schema version.")

    if analysis["symbol"] != "BTC-USDT":
        raise ValueError("Unexpected AI symbol.")

    if analysis["timeframes"] != ["4H", "15M", "5M"]:
        raise ValueError("Unexpected AI timeframe list.")

    decision = analysis["decision"]
    if decision not in ("LONG", "SHORT", "WAIT"):
        raise ValueError("Decision must be LONG, SHORT, or WAIT.")

    if analysis["market_bias"] not in ("BULLISH", "BEARISH", "NEUTRAL"):
        raise ValueError("Invalid market bias.")

    confidence = safe_float(analysis["confidence"], "confidence")
    if not 0 <= confidence <= 100:
        raise ValueError("Confidence must be between 0 and 100.")

    if not isinstance(analysis["reasoning"], str) or not analysis["reasoning"].strip():
        raise ValueError("Reasoning must be a non-empty string.")

    for field in ("main_risks", "data_limitations"):
        if not isinstance(analysis[field], list):
            raise ValueError(f"{field} must be an array.")

    entry = analysis["entry"]
    if not isinstance(entry, dict):
        raise ValueError("entry must be an object.")

    entry_type = entry.get("entry_type", entry.get("type"))
    entry_price = entry.get("entry_price", entry.get("price"))
    zone_low = entry.get("entry_zone_low")
    zone_high = entry.get("entry_zone_high")
    expiry = entry.get(
        "entry_expires_at",
        entry.get("expires_at"),
    )

    if decision == "WAIT":
        if entry_type not in (None, "NONE"):
            raise ValueError("WAIT requires NONE entry type.")

        if any(value is not None for value in (
            entry_price, zone_low, zone_high,
            analysis["stop_loss"],
            analysis["take_profit_1"],
            analysis["take_profit_2"],
            analysis["take_profit_3"],
        )):
            raise ValueError("WAIT requires null trade prices.")

    else:
        if entry_type not in ("MARKET", "LIMIT"):
            raise ValueError("LONG/SHORT requires MARKET or LIMIT entry.")

        entry_price = safe_float(entry_price, "entry price")
        stop = safe_float(analysis["stop_loss"], "stop_loss")

        targets = [
            safe_float(analysis[field], field)
            for field in (
                "take_profit_1", "take_profit_2", "take_profit_3"
            )
        ]

        validate_trade_geometry(
            decision, entry_price, stop, targets, "new trade"
        )

        if zone_low is not None:
            zone_low = safe_float(zone_low, "entry zone low")
        if zone_high is not None:
            zone_high = safe_float(zone_high, "entry zone high")

        if (zone_low is None) != (zone_high is None):
            raise ValueError("Entry zone requires both low and high.")

        if zone_low is not None and (
            zone_low > zone_high or not zone_low <= entry_price <= zone_high
        ):
            raise ValueError("Invalid entry zone.")

        if entry_type == "LIMIT":
            validate_timestamp(expiry, "entry expiry", required=True)
        elif expiry is not None:
            validate_timestamp(expiry, "entry expiry")

    previous_action = analysis["previous_signal_action"]
    if previous_action not in ("NONE", "KEEP", "REPLACE", "CANCEL"):
        raise ValueError("Invalid previous signal action.")

    if previous_signal is None and previous_action != "NONE":
        raise ValueError("No previous signal exists; action must be NONE.")

    if previous_signal is not None and previous_action == "NONE":
        raise ValueError("Previous signal exists; AI must evaluate it.")

    open_action = analysis["open_trade_action"]
    if open_action not in ("NONE", "HOLD", "CLOSE", "UPDATE_LEVELS"):
        raise ValueError("Invalid open trade action.")

    if open_trade is None and open_action != "NONE":
        raise ValueError("No open trade exists; action must be NONE.")

    if open_trade is not None and open_action == "NONE":
        raise ValueError("Open trade exists; AI must evaluate it.")

    updated = analysis["open_trade_updated_levels"]
    if not isinstance(updated, dict):
        raise ValueError("Updated levels must be an object.")

    fields = (
        "stop_loss", "take_profit_1",
        "take_profit_2", "take_profit_3",
    )

    if any(field not in updated for field in fields):
        raise ValueError("Updated levels are incomplete.")

    if open_action != "UPDATE_LEVELS":
        if any(updated[field] is not None for field in fields):
            raise ValueError("Updated levels must be null unless updating.")
    else:
        if isinstance(open_trade, list):
            if len(open_trade) != 1:
                raise ValueError("Cannot update multiple open trades.")
            trade = open_trade[0]
        else:
            trade = open_trade

        direction = str(trade.get("direction", "")).upper()
        entry_reference = safe_float(
            trade.get("entry_reference_price", trade.get("entry_price")),
            "open trade entry",
        )

        current_stop = trade.get(
            "stop_loss_current",
            trade.get("stop_loss"),
        )

        current = {
            "stop_loss": current_stop,
            "take_profit_1": trade.get("take_profit_1"),
            "take_profit_2": trade.get("take_profit_2"),
            "take_profit_3": trade.get("take_profit_3"),
        }

        resolved = {}
        for field in fields:
            proposed = updated[field]
            if proposed is None:
                proposed = current[field]
            resolved[field] = safe_float(proposed, field)

        validate_trade_geometry(
            direction,
            entry_reference,
            resolved["stop_loss"],
            [
                resolved["take_profit_1"],
                resolved["take_profit_2"],
                resolved["take_profit_3"],
            ],
            "updated open trade",
        )

    return analysis


def extract_json_object(raw_content):
    content = raw_content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    start = content.find("{")
    if start < 0:
        raise ValueError("No JSON object found in AI response.")

    result, end = json.JSONDecoder().raw_decode(content[start:])

    if content[start + end:].strip():
        raise ValueError("Unexpected text after AI JSON object.")

    return result


def run_vision_analysis(market_context, previous_signal, open_trade):
    if not CHART_FILE.is_file():
        raise FileNotFoundError(f"Vision chart not found: {CHART_FILE}")

    prompt = build_btc_analysis_prompt(
        market_context=market_context,
        previous_signal=previous_signal,
        open_trade=open_trade,
    )

    result = ask_openrouter_vision(
        image_path=CHART_FILE,
        prompt=prompt,
    )

    if not isinstance(result, dict):
        raise RuntimeError("Invalid vision provider response.")

    raw_content = result.get("content")

    if isinstance(raw_content, str) and raw_content.strip():
        atomic_write_text(VISION_RAW_FILE, raw_content)

    if result.get("status") != "success":
        raise RuntimeError(
            "Vision analysis failed: " + str(result.get("error"))
        )

    if not isinstance(raw_content, str) or not raw_content.strip():
        raise RuntimeError("AI returned empty content.")

    analysis = extract_json_object(raw_content)

    validate_vision_analysis(
        analysis, previous_signal, open_trade
    )

    analysis["analysis_id"] = str(uuid.uuid4())
    analysis["generated_at"] = utc_now().isoformat()
    analysis["model"] = result.get("model")

    return analysis


def run_daily_btc_analysis():
    now = utc_now()

    print("=" * 65)
    print("BTC MULTI-TIMEFRAME AI ANALYSIS")
    print("=" * 65)

    results = {}

    for timeframe in ("4H", "15M", "5M"):
        print(f"Loading {timeframe} market data...")
        results[timeframe] = load_timeframe_data(timeframe, now)

        freshness = results[timeframe]["freshness"]
        print(
            f"{timeframe}: {results[timeframe]['closed_candle_count']} "
            f"closed candles | age={freshness['age_minutes']}m"
        )

    stale = [
        timeframe for timeframe in ("4H", "15M", "5M")
        if not results[timeframe]["freshness"]["is_fresh"]
    ]

    if stale:
        raise RuntimeError(
            "Stale market data: " + ", ".join(stale)
        )

    latest_15m_open_ms = int(
        results["15M"]["data"].iloc[-1]["timestamp"]
    )

    latest_15m_close_ms = (
        latest_15m_open_ms + TIMEFRAME_INTERVALS_MS["15M"]
    )

    candle_id = get_candle_id(latest_15m_open_ms)

    if now.timestamp() * 1000 < latest_15m_close_ms:
        raise RuntimeError("Latest 15M candle is not closed.")

    if already_analyzed(candle_id):
        print(f"SKIPPED: 15M candle {candle_id} was already analyzed.")
        return {
            "status": "skipped",
            "reason": "duplicate_15m_candle",
            "candle_id": candle_id,
        }

    generate_4h_chart(results["4H"]["data"])

    market_context = build_multi_timeframe_context(results, now)
    previous_signal = load_previous_signal()
    open_trade = load_open_trades()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    atomic_write_json(
        CONTEXT_FILE,
        {
            "market_context": market_context,
            "previous_signal": previous_signal,
            "open_trade": open_trade,
        },
    )

    print("Sending market context to AI...")

    analysis = run_vision_analysis(
        market_context,
        previous_signal,
        open_trade,
    )

    # Publish the report only after successful AI validation.
    atomic_write_json(VISION_REPORT_FILE, analysis)

    # Save the candle checkpoint only after the report is published.
    save_successful_analysis(
        candle_id,
        analysis["analysis_id"],
    )

    print("-" * 65)
    print("AI ANALYSIS COMPLETED")
    print(f"Decision: {analysis['decision']}")
    print(f"Bias: {analysis['market_bias']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Analysis ID: {analysis['analysis_id']}")
    print(f"Report: {VISION_REPORT_FILE}")
    print("=" * 65)

    return {
        "status": "success",
        "analysis": analysis,
        "report_path": str(VISION_REPORT_FILE),
        "context_path": str(CONTEXT_FILE),
        "candle_id": candle_id,
    }


if __name__ == "__main__":
    try:
        run_daily_btc_analysis()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        raise
