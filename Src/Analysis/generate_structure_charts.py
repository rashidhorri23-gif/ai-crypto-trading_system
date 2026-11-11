
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

SWING_STYLES = {
    ("INTERNAL", "HIGH"): {
        "color": "darkred",
        "marker": "^",
        "size": 30,
        "label": "Internal Swing High",
    },
    ("INTERNAL", "LOW"): {
        "color": "darkgreen",
        "marker": "v",
        "size": 30,
        "label": "Internal Swing Low",
    },
    ("EXTERNAL", "HIGH"): {
        "color": "purple",
        "marker": "^",
        "size": 65,
        "label": "External Swing High",
    },
    ("EXTERNAL", "LOW"): {
        "color": "blue",
        "marker": "v",
        "size": 65,
        "label": "External Swing Low",
    },
}

EVENT_STYLES = {
    "BOS_BULLISH": ("green", "^"),
    "BOS_BEARISH": ("red", "v"),
    "CHOCH_BULLISH": ("blue", "P"),
    "CHOCH_BEARISH": ("orange", "X"),
    "AMBIGUOUS_DOUBLE_BREAK": ("black", "D"),
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

    candles.sort(
        key=lambda item: item["timestamp"]
    )

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
            linewidth=0.7,
            alpha=0.8,
            zorder=2
        )

        body_bottom = min(
            open_price,
            close_price
        )

        body_height = abs(
            close_price - open_price
        )

        if body_height == 0:
            body_height = max(
                abs(open_price) * 0.00005,
                0.01
            )

        candle_body = Rectangle(
            (x - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.4,
            alpha=0.8,
            zorder=2
        )

        ax.add_patch(candle_body)


def draw_all_swings(
    ax,
    structure_result,
    structure_type,
    timestamp_map,
    plotted_labels
):
    swings = structure_result.get(
        "swings",
        []
    )

    structure_type = structure_type.upper()

    for swing in swings:
        timestamp = int(
            float(swing["timestamp"])
        )

        if timestamp not in timestamp_map:
            continue

        swing_type = str(
            swing["type"]
        ).upper()

        style = SWING_STYLES.get(
            (structure_type, swing_type)
        )

        if style is None:
            continue

        label = style["label"]

        if label in plotted_labels:
            label = None
        else:
            plotted_labels.add(label)

        x = timestamp_map[timestamp]
        y = float(swing["price"])

        ax.scatter(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            s=style["size"],
            alpha=0.75,
            zorder=4,
            label=label
        )


def draw_structure_events(
    ax,
    events,
    timestamp_map,
    close_map,
    plotted_labels
):
    for event in events:
        event_type = event.get("event")

        if event_type not in EVENT_STYLES:
            continue

        timestamp = int(
            float(event["timestamp"])
        )

        if timestamp not in timestamp_map:
            continue

        color, marker = EVENT_STYLES[event_type]

        label = event_type

        if label in plotted_labels:
            label = None
        else:
            plotted_labels.add(label)

        x = timestamp_map[timestamp]
        y = float(event["level"])

        close_price = close_map.get(timestamp)

        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=75,
            zorder=6,
            label=label
        )

        if close_price is not None:
            ax.plot(
                [x, x],
                [y, close_price],
                color=color,
                linewidth=0.7,
                alpha=0.5,
                linestyle=":"
            )

    recent_events = events[-8:]

    for event in recent_events:
        timestamp = int(
            float(event["timestamp"])
        )

        if timestamp not in timestamp_map:
            continue

        event_type = event.get(
            "event",
            ""
        )

        x = timestamp_map[timestamp]
        y = float(event["level"])

        ax.annotate(
            event_type.replace("_", "\n"),
            (x, y),
            xytext=(3, 8),
            textcoords="offset points",
            fontsize=6,
            alpha=0.85
        )


def draw_latest_close(ax, candles):
    latest_close = float(
        candles[-1]["close"]
    )

    ax.axhline(
        latest_close,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.65,
        label=f"Latest Close: {latest_close:.2f}"
    )


def get_event_bias(structure_result):
    events = structure_result.get(
        "events",
        []
    )

    if events:
        last_event = events[-1]

        direction = last_event.get(
            "direction"
        )

        if direction:
            return str(direction)

        event_name = last_event.get(
            "event",
            ""
        )

        if event_name.endswith("BULLISH"):
            return "BULLISH"

        if event_name.endswith("BEARISH"):
            return "BEARISH"

    structure = structure_result.get(
        "structure",
        {}
    )

    return structure.get(
        "trend",
        "SIDEWAYS"
    )


def get_context_direction(result):
    context = result.get(
        "context",
        {}
    )

    return (
        result.get("context_direction")
        or context.get("external_trend")
        or result.get("trend")
        or "SIDEWAYS"
    )


def draw_summary(ax, result):
    internal = result["internal_structure"]
    external = result["external_structure"]

    internal_bias = get_event_bias(internal)
    external_bias = get_event_bias(external)

    context_direction = get_context_direction(
        result
    )

    summary = (
        f"Internal: {internal_bias} | "
        f"External: {external_bias}\n"
        f"Context: {context_direction} | "
        f"Directions evaluated: LONG + SHORT\n"
        f"Internal swings: {len(internal['swings'])} | "
        f"External swings: {len(external['swings'])}"
    )

    ax.text(
        0.01,
        0.99,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        }
    )


def generate_chart(timeframe):
    config = CHART_CONFIG[timeframe]

    candles = load_candles(timeframe)

    if len(candles) < 20:
        raise ValueError(
            f"Not enough candles for {timeframe}: {len(candles)}"
        )

    result = analyze_daily_structure(
        candles,
        timeframe=timeframe
    )

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

    close_map = {
        candle["timestamp"]: candle["close"]
        for candle in candles
    }

    fig, ax = plt.subplots(
        figsize=(20, 10)
    )

    draw_candles(
        ax,
        candles,
        config["candle_width"]
    )

    plotted_labels = set()

    draw_all_swings(
        ax,
        internal,
        "INTERNAL",
        timestamp_map,
        plotted_labels
    )

    draw_all_swings(
        ax,
        external,
        "EXTERNAL",
        timestamp_map,
        plotted_labels
    )

    draw_structure_events(
        ax,
        internal["events"],
        timestamp_map,
        close_map,
        plotted_labels
    )

    draw_structure_events(
        ax,
        external["events"],
        timestamp_map,
        close_map,
        plotted_labels
    )

    draw_latest_close(
        ax,
        candles
    )

    draw_summary(
        ax,
        result
    )

    ax.set_title(
        config["title"],
        fontsize=15
    )

    ax.set_ylabel("Price (USDT)")
    ax.set_xlabel("UTC Time")
    ax.grid(
        True,
        alpha=0.2
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m-%d",
            tz=timezone.utc
        )
    )

    ax.xaxis.set_major_locator(
        mdates.AutoDateLocator(
            minticks=6,
            maxticks=12
        )
    )

    fig.autofmt_xdate()

    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=2
    )

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
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"CHART_CREATED | timeframe={timeframe} "
        f"| candles={len(candles)} "
        f"| internal_bias={get_event_bias(internal)} "
        f"| external_bias={get_event_bias(external)} "
        f"| context={get_context_direction(result)} "
        f"| internal_swings={len(internal['swings'])} "
        f"| external_swings={len(external['swings'])} "
        f"| internal_events={len(internal['events'])} "
        f"| external_events={len(external['events'])} "
        f"| file={output_file}"
    )


def main():
    print("REAL OKX STRUCTURE CHART GENERATION")

    for timeframe in ("1D", "4H"):
        generate_chart(timeframe)

    print("ALL STRUCTURE CHARTS GENERATED")


if __name__ == "__main__":
    main()
