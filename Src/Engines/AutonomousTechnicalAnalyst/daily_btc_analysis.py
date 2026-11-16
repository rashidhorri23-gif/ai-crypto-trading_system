import sys
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from .market_features import prepare_market_features
from .vision_provider import ask_openrouter_vision
from .vision_prompt import build_btc_4h_prompt


ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = (
    ROOT_DIR
    / "Data"
    / "Archive"
    / "BTC_USDT_SPOT_4H.csv"
)

OUTPUT_DIR = ROOT_DIR / "Reports"

CHART_FILE = OUTPUT_DIR / "BTC_4H_Vision_Chart.png"
CONTEXT_FILE = OUTPUT_DIR / "BTC_4H_Market_Context.txt"
VISION_REPORT_FILE = OUTPUT_DIR / "BTC_4H_Vision_Report.json"
VISION_RAW_FILE = OUTPUT_DIR / "BTC_4H_Vision_Raw.txt"

CANDLE_INTERVAL_MS = 4 * 60 * 60 * 1000
MAX_CANDLES = 120
MAX_DATA_AGE_HOURS = 8

PLAN_TARGET_FIELDS = (
    "take_profit_1",
    "take_profit_2",
    "take_profit_3",
)

PLAN_RR_FIELDS = (
    "risk_reward_tp1",
    "risk_reward_tp2",
    "risk_reward_tp3",
)

PLAN_NUMERIC_FIELDS = (
    "entry_zone_low",
    "entry_zone_high",
    "stop_loss",
    *PLAN_TARGET_FIELDS,
    *PLAN_RR_FIELDS,
)


def safe_float(value, field_name, allow_none=False):
    """Validate and convert a finite numeric value."""

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


def validate_candle_intervals(df):
    """Check that all candle timestamps have exact 4H spacing."""

    differences = df["timestamp"].diff().dropna()

    invalid = differences[
        differences != CANDLE_INTERVAL_MS
    ]

    if not invalid.empty:
        raise ValueError(
            f"Invalid 4H candle intervals found: {len(invalid)}"
        )


def filter_closed_candles(df, now):
    """Exclude the currently forming 4H candle."""

    now_ms = int(now.timestamp() * 1000)

    current_candle_open = (
        now_ms // CANDLE_INTERVAL_MS
    ) * CANDLE_INTERVAL_MS

    closed = df[
        df["timestamp"] < current_candle_open
    ].copy()

    if closed.empty:
        raise ValueError(
            "No fully closed 4H candles found."
        )

    return closed


def check_data_freshness(df, now):
    """Check whether the latest closed candle is recent."""

    latest_timestamp = int(
        df.iloc[-1]["timestamp"]
    )

    latest_open_time = pd.to_datetime(
        latest_timestamp,
        unit="ms",
        utc=True,
    )

    latest_close_time = (
        latest_open_time
        + pd.Timedelta(hours=4)
    )

    age_hours = (
        now - latest_close_time.to_pydatetime()
    ).total_seconds() / 3600

    return {
        "is_fresh": (
            0 <= age_hours <= MAX_DATA_AGE_HOURS
        ),
        "age_hours": round(age_hours, 2),
        "latest_open_time": latest_open_time.isoformat(),
        "latest_close_time": latest_close_time.isoformat(),
    }


def draw_candlesticks(ax, df):
    """Draw OHLC candles using matplotlib."""

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


def generate_chart(df):
    """Generate a BTC 4H candlestick chart."""

    chart_data = (
        df.tail(MAX_CANDLES)
        .copy()
        .reset_index(drop=True)
    )

    if len(chart_data) < 50:
        raise ValueError(
            "Not enough candles to generate a reliable chart."
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

        if pd.notna(latest["support"]):
            ax.axhline(
                latest["support"],
                linestyle="--",
                linewidth=1,
                label="Latest Support",
            )

        if pd.notna(latest["resistance"]):
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
            "BTC-USDT | 4H Market Structure",
            fontsize=16,
        )

        ax.set_ylabel("Price (USDT)")
        ax.set_xlabel("UTC Time")

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            loc="upper left",
        )

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


def build_market_context(df, freshness):
    """Build numerical context for the vision model."""

    latest = df.iloc[-1]

    def safe_number(value):
        if pd.isna(value):
            return None

        return round(float(value), 4)

    context = {
        "symbol": "BTC-USDT",
        "timeframe": "4H",
        "source": "OKX",
        "latest_candle_open_utc": freshness["latest_open_time"],
        "latest_candle_close_utc": freshness["latest_close_time"],
        "data_age_hours": freshness["age_hours"],
        "data_is_fresh": freshness["is_fresh"],
        "current_price": safe_number(latest["close"]),
        "open": safe_number(latest["open"]),
        "high": safe_number(latest["high"]),
        "low": safe_number(latest["low"]),
        "close": safe_number(latest["close"]),
        "volume": safe_number(latest["volume"]),
        "ema_20": safe_number(latest["ema_20"]),
        "ema_50": safe_number(latest["ema_50"]),
        "ema_200": safe_number(latest["ema_200"]),
        "rsi_14": safe_number(latest["rsi_14"]),
        "atr_14": safe_number(latest["atr_14"]),
        "adx_14": safe_number(latest["adx_14"]),
        "plus_di": safe_number(latest["plus_di"]),
        "minus_di": safe_number(latest["minus_di"]),
        "relative_volume": safe_number(latest["relative_volume"]),
        "market_structure": str(latest["market_structure"]),
        "ema_trend": str(latest["ema_trend"]),
        "support": safe_number(latest["support"]),
        "resistance": safe_number(latest["resistance"]),
        "bullish_bos": bool(latest["bos_bullish"]),
        "bearish_bos": bool(latest["bos_bearish"]),
        "closed_candle_count": len(df),
    }

    return context


def validate_plan(plan, plan_name, direction):
    """Validate a bullish or bearish conditional trading plan."""

    if not isinstance(plan, dict):
        raise ValueError(
            f"{plan_name} must be a JSON object."
        )

    setup_type = plan.get("setup_type")

    if not isinstance(setup_type, str) or not setup_type.strip():
        raise ValueError(
            f"{plan_name}.setup_type must be a non-empty string."
        )

    activation = plan.get("activation_condition")
    invalidation = plan.get("invalidation_condition")

    if not isinstance(activation, str) or not activation.strip():
        raise ValueError(
            f"{plan_name}.activation_condition must be a non-empty string."
        )

    if not isinstance(invalidation, str) or not invalidation.strip():
        raise ValueError(
            f"{plan_name}.invalidation_condition must be a non-empty string."
        )

    if setup_type == "NONE":
        for field_name in PLAN_NUMERIC_FIELDS:
            if plan.get(field_name) is not None:
                raise ValueError(
                    f"{plan_name}: {field_name} must be null "
                    "when setup_type is NONE."
                )

        return True

    values = {}

    for field_name in PLAN_NUMERIC_FIELDS:
        values[field_name] = safe_float(
            plan.get(field_name),
            f"{plan_name}.{field_name}",
        )

    entry_low = values["entry_zone_low"]
    entry_high = values["entry_zone_high"]
    stop_loss = values["stop_loss"]

    if entry_low > entry_high:
        raise ValueError(
            f"{plan_name}: entry_zone_low cannot exceed entry_zone_high."
        )

    targets = [
        values[field_name]
        for field_name in PLAN_TARGET_FIELDS
    ]

    rr_values = [
        values[field_name]
        for field_name in PLAN_RR_FIELDS
    ]

    if any(value < 0 for value in rr_values):
        raise ValueError(
            f"{plan_name}: risk-reward values cannot be negative."
        )

    if direction == "LONG":
        if stop_loss >= entry_low:
            raise ValueError(
                f"{plan_name}: LONG stop_loss must be below entry."
            )

        if not all(
            target > entry_high for target in targets
        ):
            raise ValueError(
                f"{plan_name}: LONG targets must be above entry."
            )

        if not (
            targets[0] <= targets[1] <= targets[2]
        ):
            raise ValueError(
                f"{plan_name}: LONG targets must be ordered TP1 <= TP2 <= TP3."
            )

    elif direction == "SHORT":
        if stop_loss <= entry_high:
            raise ValueError(
                f"{plan_name}: SHORT stop_loss must be above entry."
            )

        if not all(
            target < entry_low for target in targets
        ):
            raise ValueError(
                f"{plan_name}: SHORT targets must be below entry."
            )

        if not (
            targets[0] >= targets[1] >= targets[2]
        ):
            raise ValueError(
                f"{plan_name}: SHORT targets must be ordered TP1 >= TP2 >= TP3."
            )

    else:
        raise ValueError(
            f"Unsupported plan direction: {direction}"
        )

    return True


def validate_trade_setup(trade_setup, decision):
    """Validate the final selected trade setup."""

    if not isinstance(trade_setup, dict):
        raise ValueError(
            "trade_setup must be a JSON object."
        )

    direction = trade_setup.get("direction")

    expected_direction = {
        "LONG": "LONG",
        "SHORT": "SHORT",
        "WAIT": "NONE",
    }[decision]

    if direction != expected_direction:
        raise ValueError(
            f"Decision {decision} requires trade_setup.direction "
            f"{expected_direction}, received {direction}."
        )

    for field_name in (
        "confirmation_condition",
        "invalidation_condition",
    ):
        value = trade_setup.get(field_name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"trade_setup.{field_name} must be a non-empty string."
            )

    trade_numeric_fields = [
        "entry_zone_low",
        "entry_zone_high",
        "stop_loss",
        *PLAN_TARGET_FIELDS,
        "risk_reward_ratio",
    ]

    if direction == "NONE":
        for field_name in trade_numeric_fields:
            if trade_setup.get(field_name) is not None:
                raise ValueError(
                    f"trade_setup.{field_name} must be null "
                    "when direction is NONE."
                )

        return True

    values = {}

    for field_name in trade_numeric_fields:
        values[field_name] = safe_float(
            trade_setup.get(field_name),
            f"trade_setup.{field_name}",
        )

    entry_low = values["entry_zone_low"]
    entry_high = values["entry_zone_high"]
    stop_loss = values["stop_loss"]

    targets = [
        values[field_name]
        for field_name in PLAN_TARGET_FIELDS
    ]

    rr_ratio = values["risk_reward_ratio"]

    if entry_low > entry_high:
        raise ValueError(
            "trade_setup.entry_zone_low cannot exceed entry_zone_high."
        )

    if rr_ratio < 0:
        raise ValueError(
            "trade_setup.risk_reward_ratio cannot be negative."
        )

    if direction == "LONG":
        if stop_loss >= entry_low:
            raise ValueError(
                "LONG trade_setup stop_loss must be below entry."
            )

        if not all(
            target > entry_high for target in targets
        ):
            raise ValueError(
                "LONG trade_setup targets must be above entry."
            )

        if not (
            targets[0] <= targets[1] <= targets[2]
        ):
            raise ValueError(
                "LONG trade_setup targets must be ordered TP1 <= TP2 <= TP3."
            )

    elif direction == "SHORT":
        if stop_loss <= entry_high:
            raise ValueError(
                "SHORT trade_setup stop_loss must be above entry."
            )

        if not all(
            target < entry_low for target in targets
        ):
            raise ValueError(
                "SHORT trade_setup targets must be below entry."
            )

        if not (
            targets[0] >= targets[1] >= targets[2]
        ):
            raise ValueError(
                "SHORT trade_setup targets must be ordered TP1 >= TP2 >= TP3."
            )

    return True


def validate_vision_analysis(analysis):
    """Validate the complete vision model response."""

    if not isinstance(analysis, dict):
        raise ValueError(
            "Vision response must be a JSON object."
        )

    required_fields = [
        "symbol",
        "timeframe",
        "market_bias",
        "confidence",
        "decision",
        "bullish_plan",
        "bearish_plan",
        "trade_setup",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in analysis
    ]

    if missing_fields:
        raise ValueError(
            f"Vision response is missing fields: {missing_fields}"
        )

    if analysis["symbol"] != "BTC-USDT":
        raise ValueError(
            "Vision model returned an unexpected symbol."
        )

    if analysis["timeframe"] != "4H":
        raise ValueError(
            "Vision model returned an unexpected timeframe."
        )

    if analysis["market_bias"] not in (
        "BULLISH",
        "BEARISH",
        "SIDEWAYS",
    ):
        raise ValueError(
            "Invalid market_bias. Expected BULLISH, BEARISH, or SIDEWAYS."
        )

    decision = analysis["decision"]

    if decision not in ("LONG", "SHORT", "WAIT"):
        raise ValueError(
            "Invalid decision. Expected LONG, SHORT, or WAIT."
        )

    confidence = safe_float(
        analysis["confidence"],
        "confidence",
    )

    if not 0 <= confidence <= 100:
        raise ValueError(
            "Confidence must be between 0 and 100."
        )

    if "current_price" in analysis:
        safe_float(
            analysis["current_price"],
            "current_price",
            allow_none=True,
        )

    for field_name in (
        "key_supports",
        "key_resistances",
    ):
        if field_name in analysis:
            values = analysis[field_name]

            if not isinstance(values, list):
                raise ValueError(
                    f"{field_name} must be a JSON array."
                )

            for index, value in enumerate(values):
                safe_float(
                    value,
                    f"{field_name}[{index}]",
                )

    validate_plan(
        analysis["bullish_plan"],
        "bullish_plan",
        "LONG",
    )

    validate_plan(
        analysis["bearish_plan"],
        "bearish_plan",
        "SHORT",
    )

    if decision == "LONG":
        if analysis["bullish_plan"]["setup_type"] == "NONE":
            raise ValueError(
                "LONG decision requires an active bullish_plan."
            )

    if decision == "SHORT":
        if analysis["bearish_plan"]["setup_type"] == "NONE":
            raise ValueError(
                "SHORT decision requires an active bearish_plan."
            )

    validate_trade_setup(
        analysis["trade_setup"],
        decision,
    )

    return analysis


def extract_json_object(raw_content):
    """Extract one complete JSON object from the model response."""

    cleaned_content = raw_content.strip()

    if cleaned_content.startswith("```"):
        lines = cleaned_content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_content = "\n".join(lines).strip()

    start_index = cleaned_content.find("{")

    if start_index < 0:
        raise ValueError(
            "No JSON object found in model response."
        )

    decoder = json.JSONDecoder()

    analysis, end_index = decoder.raw_decode(
        cleaned_content[start_index:]
    )

    remaining_text = (
        cleaned_content[start_index + end_index:]
    ).strip()

    if remaining_text:
        raise ValueError(
            "Unexpected text found after the JSON object."
        )

    return analysis


def run_vision_analysis(market_context):
    """Send the BTC chart and market context to a vision model."""

    if not CHART_FILE.is_file():
        raise FileNotFoundError(
            f"Vision chart not found: {CHART_FILE}"
        )

    if not market_context.get("data_is_fresh"):
        raise RuntimeError(
            "Vision analysis blocked because market data is stale."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_text = json.dumps(
        market_context,
        indent=2,
        ensure_ascii=False,
    )

    prompt = build_btc_4h_prompt(
        context_text
    )

    print("=" * 60)
    print("STARTING BTC 4H VISION ANALYSIS")
    print("=" * 60)

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
        VISION_RAW_FILE.write_text(
            raw_content,
            encoding="utf-8",
        )

        print(
            f"Raw response saved: {VISION_RAW_FILE}"
        )

    if result.get("status") != "success":
        raise RuntimeError(
            "Vision analysis failed: "
            + str(result.get("error"))
        )

    if (
        not isinstance(raw_content, str)
        or not raw_content.strip()
    ):
        raise RuntimeError(
            "Vision model returned empty content."
        )

    try:
        analysis = extract_json_object(
            raw_content
        )

        validate_vision_analysis(
            analysis
        )

    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Vision model returned invalid analysis: {error}"
        ) from error

    VISION_REPORT_FILE.write_text(
        json.dumps(
            analysis,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(f"Vision model: {result.get('model')}")
    print(f"Market bias: {analysis['market_bias']}")
    print(f"Decision: {analysis['decision']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"JSON report: {VISION_REPORT_FILE}")
    print(f"Raw response: {VISION_RAW_FILE}")
    print("VISION ANALYSIS COMPLETED.")
    print("=" * 60)

    return {
        "model": result.get("model"),
        "analysis": analysis,
        "report_path": str(VISION_REPORT_FILE),
        "raw_path": str(VISION_RAW_FILE),
    }


def run_daily_btc_analysis():
    """Prepare BTC 4H chart, context, and vision analysis."""

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Market data file not found: {DATA_FILE}"
        )

    now = datetime.now(
        timezone.utc
    )

    raw_data = pd.read_csv(
        DATA_FILE
    )

    if raw_data.empty:
        raise ValueError(
            "BTC market data file is empty."
        )

    features = prepare_market_features(
        DATA_FILE
    )

    closed_data = filter_closed_candles(
        features,
        now,
    )

    validate_candle_intervals(
        closed_data
    )

    freshness = check_data_freshness(
        closed_data,
        now,
    )

    generate_chart(
        closed_data
    )

    context = build_market_context(
        closed_data,
        freshness,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTEXT_FILE.write_text(
        json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("BTC 4H DAILY ANALYSIS")
    print("=" * 60)

    print(f"Closed candles: {len(closed_data)}")
    print(f"Latest close: {context['current_price']}")
    print(f"EMA trend: {context['ema_trend']}")
    print(f"Market structure: {context['market_structure']}")
    print(f"Data age hours: {freshness['age_hours']}")
    print(f"Data fresh: {freshness['is_fresh']}")
    print(f"Chart: {CHART_FILE}")
    print(f"Context: {CONTEXT_FILE}")

    vision_result = None

    if not freshness["is_fresh"]:
        print(
            "WARNING: Data is stale. "
            "Vision analysis will not be started."
        )

    else:
        print("DATA VALIDATION PASSED.")

        vision_result = run_vision_analysis(
            context
        )

    print("=" * 60)

    return {
        "status": (
            "ready"
            if freshness["is_fresh"]
            else "stale_data"
        ),
        "chart_path": str(CHART_FILE),
        "context_path": str(CONTEXT_FILE),
        "vision_report_path": (
            str(VISION_REPORT_FILE)
            if vision_result
            else None
        ),
        "vision_result": vision_result,
        "market_context": context,
    }


if __name__ == "__main__":
    try:
        result = run_daily_btc_analysis()

        if result["status"] != "ready":
            sys.exit(2)

    except Exception as error:
        print(
            f"ERROR: {type(error).__name__}: {error}"
        )
        sys.exit(1)
