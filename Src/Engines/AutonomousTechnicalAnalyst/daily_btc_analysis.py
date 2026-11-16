
import sys
import json
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

CHART_FILE = (
    OUTPUT_DIR
    / "BTC_4H_Vision_Chart.png"
)

CONTEXT_FILE = (
    OUTPUT_DIR
    / "BTC_4H_Market_Context.txt"
)

VISION_REPORT_FILE = (
    OUTPUT_DIR
    / "BTC_4H_Vision_Report.json"
)

VISION_RAW_FILE = (
    OUTPUT_DIR
    / "BTC_4H_Vision_Raw.txt"
)

CANDLE_INTERVAL_MS = 4 * 60 * 60 * 1000
MAX_CANDLES = 120
MAX_DATA_AGE_HOURS = 8


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

    chart_data = df.tail(
        MAX_CANDLES
    ).copy().reset_index(drop=True)

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

    if result.get("status") != "success":
        raise RuntimeError(
            "Vision analysis failed: "
            + str(result.get("error"))
        )

    raw_content = result.get("content")

    if (
        not isinstance(raw_content, str)
        or not raw_content.strip()
    ):
        raise RuntimeError(
            "Vision model returned empty content."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISION_RAW_FILE.write_text(
        raw_content,
        encoding="utf-8",
    )

    cleaned_content = raw_content.strip()

    if cleaned_content.startswith("```"):
        lines = cleaned_content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_content = "\n".join(lines).strip()

    try:
        decoder = json.JSONDecoder()

        start_index = cleaned_content.find("{")

        if start_index < 0:
            raise ValueError(
                "No JSON object found in model response."
            )

        analysis, _ = decoder.raw_decode(
            cleaned_content[start_index:]
        )

    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Vision model returned invalid JSON: {error}"
        )

    if not isinstance(analysis, dict):
        raise RuntimeError(
            "Vision response must be a JSON object."
        )

    required_fields = [
        "symbol",
        "timeframe",
        "market_bias",
        "confidence",
        "decision",
        "trade_setup",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in analysis
    ]

    if missing_fields:
        raise RuntimeError(
            f"Vision response is missing fields: {missing_fields}"
        )

    if analysis["symbol"] != "BTC-USDT":
        raise RuntimeError(
            "Vision model returned an unexpected symbol."
        )

    if analysis["timeframe"] != "4H":
        raise RuntimeError(
            "Vision model returned an unexpected timeframe."
        )

    if analysis["decision"] not in [
        "LONG",
        "SHORT",
        "WAIT",
    ]:
        raise RuntimeError(
            "Invalid trading decision returned by vision model."
        )

    if analysis["market_bias"] not in [
        "BULLISH",
        "BEARISH",
        "SIDEWAYS",
    ]:
        raise RuntimeError(
            "Invalid market bias returned by vision model."
        )

    confidence = analysis["confidence"]

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 100
    ):
        raise RuntimeError(
            "Confidence must be a number between 0 and 100."
        )

    trade_setup = analysis["trade_setup"]

    if not isinstance(trade_setup, dict):
        raise RuntimeError(
            "Invalid trade_setup structure."
        )

    if trade_setup.get("direction") not in [
        "LONG",
        "SHORT",
        "NONE",
    ]:
        raise RuntimeError(
            "Invalid trade setup direction."
        )

    if (
        analysis["decision"] == "WAIT"
        and trade_setup["direction"] != "NONE"
    ):
        raise RuntimeError(
            "WAIT decision must have trade_setup direction NONE."
        )

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
