
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


def atomic_write_json(path, data):
    """Write JSON atomically to avoid publishing partial files."""

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def json_safe(value):
    """Convert pandas and numeric values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = value

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        return timestamp.astimezone(
            timezone.utc
        ).isoformat()

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
    """Validate a finite numeric value."""

    if value is None and allow_none:
        return None

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric, not boolean."
        )

    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be numeric."
        )

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be finite."
        )

    return number


def normalize_market_data(df, timeframe):
    """Sort and validate the candle data without inventing candles."""

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"{timeframe}: missing candle columns: {sorted(missing)}"
        )

    data = df.copy()

    data["timestamp"] = pd.to_numeric(
        data["timestamp"],
        errors="raise",
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data["timestamp"].isna().any():
        raise ValueError(
            f"{timeframe}: candle timestamps contain null values."
        )

    if data["timestamp"].duplicated().any():
        raise ValueError(
            f"{timeframe}: duplicate candle timestamps detected."
        )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            f"{timeframe}: candle data is empty."
        )

    if not data["timestamp"].apply(
        lambda value: math.isfinite(float(value))
    ).all():
        raise ValueError(
            f"{timeframe}: invalid candle timestamps."
        )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        if not data[column].apply(
            lambda value: math.isfinite(float(value))
        ).all():
            raise ValueError(
                f"{timeframe}: non-finite values in {column}."
            )

    if (
        (data[["open", "high", "low", "close"]] <= 0)
        .any()
        .any()
    ):
        raise ValueError(
            f"{timeframe}: prices must be positive."
        )

    if (data["volume"] < 0).any():
        raise ValueError(
            f"{timeframe}: volume cannot be negative."
        )

    invalid_ohlc = (
        (data["high"] < data["low"])
        | (data["high"] < data["open"])
        | (data["high"] < data["close"])
        | (data["low"] > data["open"])
        | (data["low"] > data["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            f"{timeframe}: invalid OHLC candle geometry."
        )

    return data


def filter_closed_candles(df, timeframe, now):
    """Exclude the currently forming candle for a timeframe."""

    interval_ms = TIMEFRAME_INTERVALS_MS[timeframe]
    now_ms = int(now.timestamp() * 1000)

    current_candle_open = (
        now_ms // interval_ms
    ) * interval_ms

    closed = df[
        df["timestamp"] < current_candle_open
    ].copy()

    if closed.empty:
        raise ValueError(
            f"{timeframe}: no fully closed candles found."
        )

    return closed.reset_index(drop=True)


def get_data_freshness(df, timeframe, now):
    """Calculate freshness from the latest confirmed candle close."""

    interval_ms = TIMEFRAME_INTERVALS_MS[timeframe]

    latest_timestamp = int(
        df.iloc[-1]["timestamp"]
    )

    latest_open = pd.to_datetime(
        latest_timestamp,
        unit="ms",
        utc=True,
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
        "is_fresh": (
            0 <= age_minutes <= max_age
        ),
    }


def get_interval_gaps(df, timeframe):
    """Report missing intervals without silently filling them."""

    interval_ms = TIMEFRAME_INTERVALS_MS[timeframe]

    differences = df["timestamp"].diff().dropna()

    gaps = differences[
        differences > interval_ms
    ]

    return {
        "missing_interval_count": int(len(gaps)),
        "largest_interval_ms": (
            int(gaps.max())
            if not gaps.empty
            else interval_ms
        ),
    }


def load_timeframe_data(timeframe, now):
    """Load, feature-engineer, and validate one timeframe."""

    file_path = DATA_FILES[timeframe]

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{timeframe} archive not found: {file_path}"
        )

    raw_data = pd.read_csv(
        file_path
    )

    raw_data = normalize_market_data(
        raw_data,
        timeframe,
    )

    features = prepare_market_features(
        file_path
    )

    features = normalize_market_data(
        features,
        timeframe,
    )

    closed = filter_closed_candles(
        features,
        timeframe,
        now,
    )

    if len(closed) < 30:
        raise ValueError(
            f"{timeframe}: insufficient confirmed candle history."
        )

    freshness = get_data_freshness(
        closed,
        timeframe,
        now,
    )

    gaps = get_interval_gaps(
        closed,
        timeframe,
    )

    return {
        "data": closed,
        "freshness": freshness,
        "gaps": gaps,
        "raw_candle_count": len(raw_data),
        "closed_candle_count": len(closed),
    }


def draw_candlesticks(ax, df):
    """Draw OHLC candles."""

    candle_width = 0.65

    for index, row in enumerate(
        df.itertuples(index=False)
    ):
        bullish = row.close >= row.open

        candle_color = (
            "green" if bullish else "red"
        )

        ax.vlines(
            index,
            row.low,
            row.high,
            color=candle_color,
            linewidth=0.8,
        )

        body_bottom = min(
            row.open,
            row.close,
        )

        body_height = abs(
            row.close - row.open
        )

        if body_height == 0:
            body_height = max(
                row.close * 0.0001,
                0.01,
            )

        candle = Rectangle(
            (
                index - candle_width / 2,
                body_bottom,
            ),
            candle_width,
            body_height,
            facecolor=candle_color,
            edgecolor=candle_color,
            linewidth=0.7,
        )

        ax.add_patch(candle)


def generate_4h_chart(df):
    """Generate the 4H chart used as visual context by the AI."""

    chart_data = (
        df.tail(MAX_CONTEXT_CANDLES["4H"])
        .copy()
        .reset_index(drop=True)
    )

    if len(chart_data) < 50:
        raise ValueError(
            "Not enough 4H candles to generate the chart."
        )

    required_features = (
        "ema_20",
        "ema_50",
        "ema_200",
        "timestamp_utc",
    )

    missing = [
        field
        for field in required_features
        if field not in chart_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing chart features: {missing}"
        )

    fig, ax = plt.subplots(
        figsize=(18, 10)
    )

    try:
        draw_candlesticks(
            ax,
            chart_data,
        )

        ax.plot(
            chart_data.index,
            chart_data["ema_20"],
            label="EMA 20",
            linewidth=1.2,
        )

        ax.plot(
            chart_data.index,
            chart_data["ema_50"],
            label="EMA 50",
            linewidth=1.2,
        )

        ax.plot(
            chart_data.index,
            chart_data["ema_200"],
            label="EMA 200",
            linewidth=1.2,
        )

        latest = chart_data.iloc[-1]

        if (
            "support" in chart_data.columns
            and pd.notna(latest["support"])
        ):
            ax.axhline(
                latest["support"],
                linestyle="--",
                linewidth=1,
                label="Latest Support",
            )

        if (
            "resistance" in chart_data.columns
            and pd.notna(latest["resistance"])
        ):
            ax.axhline(
                latest["resistance"],
                linestyle="--",
                linewidth=1,
                label="Latest Resistance",
            )

        tick_positions = list(
            range(
                0,
                len(chart_data),
                max(1, len(chart_data) // 12),
            )
        )

        tick_labels = [
            chart_data.iloc[index][
                "timestamp_utc"
            ].strftime("%m-%d %H:%M")
            for index in tick_positions
        ]

        ax.set_xticks(tick_positions)

        ax.set_xticklabels(
            tick_labels,
            rotation=35,
            ha="right",
        )

        ax.set_title(
            "BTC-USDT | 4H Market Context",
            fontsize=16,
        )

        ax.set_ylabel("Price (USDT)")
        ax.set_xlabel("UTC Time")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            CHART_FILE,
            dpi=180,
            bbox_inches="tight",
        )

    finally:
        plt.close(fig)

    return chart_data


def build_timeframe_context(timeframe, timeframe_result):
    """Build structured numerical context for one timeframe."""

    df = timeframe_result["data"]
    latest = df.iloc[-1]

    excluded_columns = {
        "timestamp_utc",
    }

    candle_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    recent_candles = []

    for _, row in df.tail(
        MAX_CONTEXT_CANDLES[timeframe]
    ).iterrows():
        candle = {
            column: json_safe(row[column])
            for column in candle_columns
        }

        candle["timestamp_utc"] = (
            pd.to_datetime(
                int(row["timestamp"]),
                unit="ms",
                utc=True,
            ).isoformat()
        )

        recent_candles.append(candle)

    return {
        "timeframe": timeframe,
        "source": "OKX",
        "freshness": timeframe_result["freshness"],
        "data_quality": timeframe_result["gaps"],
        "raw_candle_count": timeframe_result["raw_candle_count"],
        "closed_candle_count": timeframe_result["closed_candle_count"],
        "latest_confirmed_candle": recent_candles[-1],
        "recent_confirmed_candles": recent_candles,
        "latest_close": json_safe(latest["close"]),
    }


def load_previous_signal():
    """Load the previously published signal when available."""

    if not CURRENT_SIGNAL_FILE.is_file():
        return None

    try:
        signal = json.loads(
            CURRENT_SIGNAL_FILE.read_text(
                encoding="utf-8"
            )
        )
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(
            f"Cannot read previous signal: {error}"
        ) from error

    if not isinstance(signal, dict):
        raise ValueError(
            "Previous signal file must contain a JSON object."
        )

    return signal


def load_open_trades():
    """Read open paper trades using the paper-trading module."""

    ledger = load_ledger()

    if not isinstance(ledger, dict):
        raise ValueError(
            "Paper-trading ledger must be a JSON object."
        )

    trades = ledger.get("trades", [])

    if not isinstance(trades, list):
        raise ValueError(
            "Paper-trading ledger trades must be a list."
        )

    open_trades = [
        trade
        for trade in trades
        if isinstance(trade, dict)
        and str(trade.get("status", "")).upper() == "OPEN"
    ]

    if not open_trades:
        return None

    if len(open_trades) == 1:
        return open_trades[0]

    return open_trades


def build_multi_timeframe_context(timeframe_results, now):
    """Combine all timeframe evidence into one AI input object."""

    timeframe_contexts = {
        timeframe: build_timeframe_context(
            timeframe,
            timeframe_results[timeframe],
        )
        for timeframe in ("4H", "15M", "5M")
    }

    return {
        "symbol": "BTC-USDT",
        "source": "OKX",
        "analysis_trigger": "CLOSED_15M_CANDLE",
        "analysis_time_utc": now.isoformat(),
        "timeframes": timeframe_contexts,
        "data_status": {
            timeframe: timeframe_results[timeframe]["freshness"]
            for timeframe in ("4H", "15M", "5M")
        },
        "data_quality": {
            timeframe: timeframe_results[timeframe]["gaps"]
            for timeframe in ("4H", "15M", "5M")
        },
    }


def validate_timestamp(value, field_name, required=False):
    """Validate an ISO 8601 UTC timestamp."""

    if value is None:
        if required:
            raise ValueError(
                f"{field_name} is required."
            )
        return None

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be an ISO 8601 timestamp."
        )

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            f"{field_name} is not a valid ISO 8601 timestamp."
        ) from error

    if parsed.tzinfo is None:
        raise ValueError(
            f"{field_name} must include a UTC timezone."
        )

    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(
            f"{field_name} must use UTC."
        )

    return parsed


def validate_trade_geometry(
    direction,
    entry_price,
    stop_loss,
    targets,
    field_prefix,
):
    """Validate directional trade-price geometry."""

    if direction == "LONG":
        if stop_loss >= entry_price:
            raise ValueError(
                f"{field_prefix}: LONG stop loss must be below entry."
            )

        if not all(
            target > entry_price
            for target in targets
        ):
            raise ValueError(
                f"{field_prefix}: LONG targets must be above entry."
            )

        if not (
            targets[0] <= targets[1] <= targets[2]
        ):
            raise ValueError(
                f"{field_prefix}: LONG targets must be ordered."
            )

    elif direction == "SHORT":
        if stop_loss <= entry_price:
            raise ValueError(
                f"{field_prefix}: SHORT stop loss must be above entry."
            )

        if not all(
            target < entry_price
            for target in targets
        ):
            raise ValueError(
                f"{field_prefix}: SHORT targets must be below entry."
            )

        if not (
            targets[0] >= targets[1] >= targets[2]
        ):
            raise ValueError(
                f"{field_prefix}: SHORT targets must be ordered."
            )

    else:
        raise ValueError(
            f"{field_prefix}: unsupported direction."
        )


def validate_vision_analysis(analysis, previous_signal, open_trade):
    """Validate the AI response against schema version 2.0."""

    if not isinstance(analysis, dict):
        raise ValueError(
            "AI response must be a JSON object."
        )

    required_fields = {
        "schema_version",
        "symbol",
        "timeframes",
        "decision",
        "confidence",
        "market_bias",
        "entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "take_profit_3",
        "reasoning",
        "main_risks",
        "data_limitations",
        "previous_signal_action",
        "open_trade_action",
        "open_trade_updated_levels",
    }

    missing = required_fields.difference(
        analysis.keys()
    )

    if missing:
        raise ValueError(
            f"AI response is missing fields: {sorted(missing)}"
        )

    if analysis["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported AI response schema version."
        )

    if analysis["symbol"] != "BTC-USDT":
        raise ValueError(
            "AI returned an unexpected symbol."
        )

    if analysis["timeframes"] != ["4H", "15M", "5M"]:
        raise ValueError(
            "AI returned an unexpected timeframe list."
        )

    decision = analysis["decision"]

    if decision not in ("LONG", "SHORT", "WAIT"):
        raise ValueError(
            "decision must be LONG, SHORT, or WAIT."
        )

    if analysis["market_bias"] not in (
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ):
        raise ValueError(
            "market_bias must be BULLISH, BEARISH, or NEUTRAL."
        )

    confidence = safe_float(
        analysis["confidence"],
        "confidence",
    )

    if not 0 <= confidence <= 100:
        raise ValueError(
            "confidence must be between 0 and 100."
        )

    if not isinstance(analysis["reasoning"], str) or not analysis["reasoning"].strip():
        raise ValueError(
            "reasoning must be a non-empty string."
        )

    for field_name in ("main_risks", "data_limitations"):
        if not isinstance(analysis[field_name], list):
            raise ValueError(
                f"{field_name} must be a JSON array."
            )

    entry = analysis["entry"]

    if not isinstance(entry, dict):
        raise ValueError(
            "entry must be a JSON object."
        )

    entry_fields = {
        "entry_type",
        "entry_price",
        "entry_zone_low",
        "entry_zone_high",
        "entry_expires_at",
    }

    if not entry_fields.issubset(entry):
        raise ValueError(
            "entry is missing required fields."
        )

    entry_type = entry["entry_type"]

    if entry_type not in ("MARKET", "LIMIT", "NONE"):
        raise ValueError(
            "entry_type must be MARKET, LIMIT, or NONE."
        )

    price_fields = (
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "take_profit_3",
    )

    if decision == "WAIT":
        if entry_type != "NONE":
            raise ValueError(
                "WAIT requires entry_type NONE."
            )

        if any(
            entry.get(field) is not None
            for field in (
                "entry_price",
                "entry_zone_low",
                "entry_zone_high",
            )
        ):
            raise ValueError(
                "WAIT requires null entry prices."
            )

        if any(
            analysis.get(field) is not None
            for field in price_fields
        ):
            raise ValueError(
                "WAIT requires null stop loss and targets."
            )

        if entry.get("entry_expires_at") is not None:
            raise ValueError(
                "WAIT requires null entry expiry."
            )

    else:
        if entry_type not in ("MARKET", "LIMIT"):
            raise ValueError(
                "LONG and SHORT require MARKET or LIMIT entry."
            )

        entry_price = safe_float(
            entry.get("entry_price"),
            "entry.entry_price",
        )

        entry_low = safe_float(
            entry.get("entry_zone_low"),
            "entry.entry_zone_low",
        )

        entry_high = safe_float(
            entry.get("entry_zone_high"),
            "entry.entry_zone_high",
        )

        if entry_low > entry_high:
            raise ValueError(
                "Entry zone low cannot exceed entry zone high."
            )

        if not entry_low <= entry_price <= entry_high:
            raise ValueError(
                "Entry price must be inside the entry zone."
            )

        stop_loss = safe_float(
            analysis["stop_loss"],
            "stop_loss",
        )

        targets = [
            safe_float(
                analysis[field],
                field,
            )
            for field in (
                "take_profit_1",
                "take_profit_2",
                "take_profit_3",
            )
        ]

        validate_trade_geometry(
            decision,
            entry_price,
            stop_loss,
            targets,
            "new trade",
        )

        if entry_type == "LIMIT":
            validate_timestamp(
                entry.get("entry_expires_at"),
                "entry.entry_expires_at",
                required=True,
            )

        elif entry.get("entry_expires_at") is not None:
            validate_timestamp(
                entry.get("entry_expires_at"),
                "entry.entry_expires_at",
            )

    previous_action = analysis["previous_signal_action"]

    if previous_action not in (
        "NONE",
        "KEEP",
        "REPLACE",
        "CANCEL",
    ):
        raise ValueError(
            "Invalid previous_signal_action."
        )

    if previous_signal is None:
        if previous_action != "NONE":
            raise ValueError(
                "No previous signal exists; previous_signal_action must be NONE."
            )
    elif previous_action == "NONE":
        raise ValueError(
            "A previous signal exists; AI must evaluate it."
        )

    open_action = analysis["open_trade_action"]

    if open_action not in (
        "NONE",
        "HOLD",
        "CLOSE",
        "UPDATE_LEVELS",
    ):
        raise ValueError(
            "Invalid open_trade_action."
        )

    if open_trade is None:
        if open_action != "NONE":
            raise ValueError(
                "No open trade exists; open_trade_action must be NONE."
            )
    elif open_action == "NONE":
        raise ValueError(
            "An open trade exists; AI must evaluate it."
        )

    updated_levels = analysis["open_trade_updated_levels"]

    if not isinstance(updated_levels, dict):
        raise ValueError(
            "open_trade_updated_levels must be a JSON object."
        )

    update_fields = (
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "take_profit_3",
    )

    if not all(
        field in updated_levels
        for field in update_fields
    ):
        raise ValueError(
            "open_trade_updated_levels is missing fields."
        )

    if open_action != "UPDATE_LEVELS":
        if any(
            updated_levels[field] is not None
            for field in update_fields
        ):
            raise ValueError(
                "Updated levels must be null unless action is UPDATE_LEVELS."
            )

    else:
        if open_trade is None:
            raise ValueError(
                "Cannot update levels without an open trade."
            )

        if isinstance(open_trade, list):
            if len(open_trade) != 1:
                raise ValueError(
                    "UPDATE_LEVELS is not supported for multiple open trades in one response."
                )
            trade = open_trade[0]
        else:
            trade = open_trade

        direction = str(
            trade.get("direction", "")
        ).upper()

        entry_price = safe_float(
            trade.get("entry_reference_price", trade.get("entry_price")),
            "open_trade.entry_price",
        )

        current_levels = {
            "stop_loss": trade.get("stop_loss"),
            "take_profit_1": trade.get("take_profit_1"),
            "take_profit_2": trade.get("take_profit_2"),
            "take_profit_3": trade.get("take_profit_3"),
        }

        resolved_levels = {}

        for field in update_fields:
            proposed = updated_levels[field]

            if proposed is None:
                proposed = current_levels[field]

            resolved_levels[field] = safe_float(
                proposed,
                f"open_trade_updated_levels.{field}",
            )

        validate_trade_geometry(
            direction,
            entry_price,
            resolved_levels["stop_loss"],
            [
                resolved_levels["take_profit_1"],
                resolved_levels["take_profit_2"],
                resolved_levels["take_profit_3"],
            ],
            "updated open trade",
        )

    return analysis


def extract_json_object(raw_content):
    """Extract exactly one JSON object from the model response."""

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
        raise ValueError(
            "No JSON object found in AI response."
        )

    decoder = json.JSONDecoder()

    result, end = decoder.raw_decode(
        content[start:]
    )

    trailing = content[start + end:].strip()

    if trailing:
        raise ValueError(
            "Unexpected text found after the JSON object."
        )

    return result


def run_vision_analysis(
    market_context,
    previous_signal,
    open_trade,
):
    """Ask the AI for one multi-timeframe decision."""

    if not CHART_FILE.is_file():
        raise FileNotFoundError(
            f"Vision chart not found: {CHART_FILE}"
        )

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
        raise RuntimeError(
            "Invalid response from vision provider."
        )

    raw_content = result.get("content")

    if isinstance(raw_content, str) and raw_content.strip():
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        VISION_RAW_FILE.write_text(
            raw_content,
            encoding="utf-8",
        )

    if result.get("status") != "success":
        raise RuntimeError(
            "Vision analysis failed: "
            + str(result.get("error"))
        )

    if not isinstance(raw_content, str) or not raw_content.strip():
        raise RuntimeError(
            "AI returned empty content."
        )

    analysis = extract_json_object(
        raw_content
    )

    validate_vision_analysis(
        analysis,
        previous_signal,
        open_trade,
    )

    now = utc_now()

    # These metadata fields are controlled by Python.
    analysis["analysis_id"] = str(uuid.uuid4())
    analysis["generated_at"] = now.isoformat()
    analysis["model"] = result.get("model")

    return analysis


def run_daily_btc_analysis():
    """Run one BTC analysis after a confirmed 15M candle."""

    now = utc_now()

    print("=" * 65)
    print("BTC MULTI-TIMEFRAME AI ANALYSIS")
    print("=" * 65)

    timeframe_results = {}

    for timeframe in ("4H", "15M", "5M"):
        print(f"Loading {timeframe} market data...")

        timeframe_results[timeframe] = load_timeframe_data(
            timeframe,
            now,
        )

        freshness = timeframe_results[timeframe]["freshness"]

        print(
            f"{timeframe}: "
            f"{timeframe_results[timeframe]['closed_candle_count']} "
            f"closed candles | "
            f"age={freshness['age_minutes']} minutes | "
            f"fresh={freshness['is_fresh']}"
        )

    stale_timeframes = [
        timeframe
        for timeframe in ("4H", "15M", "5M")
        if not timeframe_results[timeframe]["freshness"]["is_fresh"]
    ]

    if stale_timeframes:
        raise RuntimeError(
            "Analysis stopped because these timeframes are stale: "
            + ", ".join(stale_timeframes)
        )

    # The 15M archive must contain a newly closed candle.
    latest_15m_close = pd.to_datetime(
        int(timeframe_results["15M"]["data"].iloc[-1]["timestamp"]),
        unit="ms",
        utc=True,
    ) + pd.Timedelta(minutes=15)

    latest_15m_close_dt = latest_15m_close.to_pydatetime()

    if (now - latest_15m_close_dt).total_seconds() < 0:
        raise RuntimeError(
            "Latest 15M candle close is in the future."
        )

    generate_4h_chart(
        timeframe_results["4H"]["data"]
    )

    market_context = build_multi_timeframe_context(
        timeframe_results,
        now,
    )

    previous_signal = load_previous_signal()
    open_trade = load_open_trades()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_json(
        CONTEXT_FILE,
        {
            "market_context": market_context,
            "previous_signal": previous_signal,
            "open_trade": open_trade,
        },
    )

    print("All timeframe data passed freshness checks.")
    print(
        "Previous signal:",
        "available" if previous_signal is not None else "none",
    )
    print(
        "Open paper trade:",
        "available" if open_trade is not None else "none",
    )
    print("Sending market context to AI...")

    analysis = run_vision_analysis(
        market_context=market_context,
        previous_signal=previous_signal,
        open_trade=open_trade,
    )

    # Publish only after the complete AI response passes validation.
    atomic_write_json(
        VISION_REPORT_FILE,
        analysis,
    )

    print("-" * 65)
    print("AI ANALYSIS COMPLETED")
    print(f"Decision: {analysis['decision']}")
    print(f"Market bias: {analysis['market_bias']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Analysis ID: {analysis['analysis_id']}")
    print(f"Generated at: {analysis['generated_at']}")
    print(f"Report: {VISION_REPORT_FILE}")
    print("=" * 65)

    return {
        "status": "success",
        "analysis": analysis,
        "report_path": str(VISION_REPORT_FILE),
        "context_path": str(CONTEXT_FILE),
    }


if __name__ == "__main__":
    try:
        run_daily_btc_analysis()

    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}"
        )
        raise
