import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from Src.Analysis.daily_structure import analyze_daily_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"
OUTPUT_DIR = PROJECT_ROOT / "Reports"

CHART_CONFIG = {
    "1D": {
        "file": "BTC_USDT_SPOT_1D.csv",
        "limit": 90,
        "title": "BTC-USDT Daily Market Structure",
        "candle_width": 0.6,
    },
    "4H": {
        "file": "BTC_USDT_SPOT_4H.csv",
        "limit": 540,
        "title": "BTC-USDT 4H Market Structure",
        "candle_width": 0.12,
    },
}


def to_datetime(timestamp):
    timestamp = int(float(timestamp))

    if timestamp > 100000000000:
        timestamp //= 1000

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    )


def load_candles(timeframe):
    config = CHART_CONFIG[timeframe]
    archive_file = ARCHIVE_DIR / config["file"]

    if not archive_file.exists():
        raise FileNotFoundError(
            f"Archive file not found: {archive_file}"
        )

    candles = []

    with archive_file.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            candles.append({
                "timestamp": int(float(row["timestamp"])),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

    candles.sort(key=lambda item: item["timestamp"])

    return candles[-config["limit"]:]


def draw_candles(ax, candles, candle_width):
    for candle in candles:
        x = mdates.date2num(
            to_datetime(candle["timestamp"])
        )

        open_price = candle["open"]
        high_price = candle["high"]
        low_price = candle["low"]
        close_price = candle["close"]

        color = (
            "green"
            if close_price >= open_price
            else "red"
        )

        ax.vlines(
            x,
            low_price,
            high_price,
            color=color,
            linewidth=0.8,
            alpha=0.9
        )

        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)

        if body_height == 0:
            body_height = max(abs(open_price) * 0.00005, 0.01)

        candle_body = Rectangle(
            (x - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            alpha=0.8
        )

        ax.add_patch(candle_body)


def draw_swing_points(ax, structure, timestamp_map):
    swing_items = [
        ("previous_high", "Previous Swing High", "red", "^"),
        ("latest_high", "Latest Swing High", "darkred", "^"),
        ("previous_low", "Previous Swing Low", "green", "v"),
        ("latest_low", "Latest Swing Low", "darkgreen", "v"),
    ]

    for key, label, color, marker in swing_items:
        point = structure.get(key)

        if not point:
            continue

        timestamp = int(float(point["timestamp"]))

        if timestamp not in timestamp_map:
            continue

        x = timestamp_map[timestamp]
        y = float(point["price"])

        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=75,
            zorder=5,
            label=label
        )


def draw_structure_events(ax, events, timestamp_map):
    styles = {
        "BOS_BULLISH": ("green", "^"),
        "BOS_BEARISH": ("red", "v"),
        "CHOCH_BULLISH": ("blue", "P"),
        "CHOCH_BEARISH": ("orange", "X"),
    }

    plotted_labels = set()

    for event in events:
        event_type = event.get("event")
        timestamp = int(float(event["timestamp"]))

        if timestamp not in timestamp_map:
            continue

        if event_type not in styles:
            continue

        color, marker = styles[event_type]
        label = event_type if event_type not in plotted_labels else None

        ax.scatter(
            timestamp_map[timestamp],
            float(event["level"]),
            color=color,
            marker=marker,
            s=65,
            zorder=6,
            label=label
        )

        plotted_labels.add(event_type)

    recent_events = events[-8:]

    for event in recent_events:
        timestamp = int(float(event["timestamp"]))

        if timestamp not in timestamp_map:
            continue

        event_type = event.get("event", "")
        x = timestamp_map[timestamp]
        y = float(event["level"])

        ax.annotate(
            event_type.replace("_", "\n"),
            (x, y),
            xytext=(4, 9),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85
        )


def generate_chart(timeframe):
    config = CHART_CONFIG[timeframe]
    candles = load_candles(timeframe)

    if len(candles) < 20:
        raise ValueError(
            f"Not enough candles for {timeframe}: {len(candles)}"
        )

    result = analyze_daily_structure(candles)

    internal = result["internal_structure"]
    external = result["external_structure"]

    dates = [
        to_datetime(candle["timestamp"])
        for candle in candles
    ]

    timestamp_map = {
        candle["timestamp"]: mdates.date2num(date)
        for candle, date in zip(candles, dates)
    }

    fig, ax = plt.subplots(figsize=(18, 9))

    draw_candles(
        ax,
        candles,
        config["candle_width"]
    )

    draw_swing_points(
        ax,
        internal.get("structure", {}),
        timestamp_map
    )

    draw_structure_events(
        ax,
        internal.get("events", []),
        timestamp_map
    )

    latest_close = candles[-1]["close"]

    ax.axhline(
        latest_close,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.65,
        label=f"Latest Close: {latest_close:.2f}"
    )

    ax.set_title(
        f"{config['title']}\n"
        f"Internal: {internal.get('trend')} | "
        f"External: {external.get('trend')} | "
        f"Allowed: {result.get('allowed_direction')}",
        fontsize=14
    )

    ax.set_ylabel("Price (USDT)")
    ax.set_xlabel("UTC Time")
    ax.grid(True, alpha=0.2)

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m-%d",
            tz=timezone.utc
        )
    )

    fig.autofmt_xdate()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / (
        f"BTC_{timeframe}_Structure.png"
    )

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"CHART_CREATED | timeframe={timeframe} "
        f"| candles={len(candles)} "
        f"| internal={internal.get('trend')} "
        f"| external={external.get('trend')} "
        f"| direction={result.get('allowed_direction')} "
        f"| file={output_file}"
    )


def main():
    print("REAL OKX STRUCTURE CHART GENERATION")

    for timeframe in ("1D", "4H"):
        generate_chart(timeframe)

    print("ALL STRUCTURE CHARTS GENERATED")


if __name__ == "__main__":
    main()
