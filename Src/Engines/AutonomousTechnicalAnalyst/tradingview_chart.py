"""TradingView Lightweight Charts renderer for the BTC multi-timeframe report."""

import json
import math
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright


LIGHTWEIGHT_CHARTS_VERSION = "5.3.0"
CHART_CANDLE_LIMIT = 120
CHART_HEIGHT = 430

TIMEFRAME_LABELS = {
    "4H": "Higher-timeframe structure",
    "15M": "Intermediate structure",
    "5M": "Execution context",
}

TIMEFRAME_SECONDS = {
    "4H": 4 * 60 * 60,
    "15M": 15 * 60,
    "5M": 5 * 60,
}

SWING_CONFIRMATION_CANDLES = 3


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _boolean(value):
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _prepare_timeframe(timeframe, result):
    frame = result["data"].tail(CHART_CANDLE_LIMIT).copy()
    if frame.empty:
        raise ValueError(f"{timeframe}: no candles available for chart.")

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{timeframe}: missing chart columns: {sorted(missing)}"
        )

    candles = []
    ema = {"ema_20": [], "ema_50": [], "ema_200": []}
    volume = []
    markers = []
    chart_start_time = int(
        _number(frame.iloc[0]["timestamp"]) // 1000
    )

    for row in frame.to_dict(orient="records"):
        timestamp = _number(row.get("timestamp"))
        if timestamp is None:
            raise ValueError(f"{timeframe}: invalid chart timestamp.")

        candle_time = int(timestamp // 1000)
        open_price = _number(row.get("open"))
        high_price = _number(row.get("high"))
        low_price = _number(row.get("low"))
        close_price = _number(row.get("close"))
        candle_volume = _number(row.get("volume"))

        if None in (open_price, high_price, low_price, close_price, candle_volume):
            raise ValueError(f"{timeframe}: invalid OHLCV chart values.")

        candles.append({
            "time": candle_time,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
        })

        for field in ema:
            value = _number(row.get(field))
            if value is not None:
                ema[field].append({"time": candle_time, "value": value})

        volume.append({
            "time": candle_time,
            "value": candle_volume,
            "color": (
                "rgba(8, 153, 129, 0.48)"
                if close_price >= open_price
                else "rgba(242, 54, 69, 0.48)"
            ),
        })

        swing_high = _number(row.get("swing_high"))
        swing_low = _number(row.get("swing_low"))
        pivot_time = (
            candle_time
            - SWING_CONFIRMATION_CANDLES * TIMEFRAME_SECONDS[timeframe]
        )
        if swing_high is not None and pivot_time >= chart_start_time:
            markers.append({
                "time": pivot_time,
                "position": "aboveBar",
                "color": "#f59e0b",
                "shape": "circle",
                "text": "SH",
            })
        if swing_low is not None and pivot_time >= chart_start_time:
            markers.append({
                "time": pivot_time,
                "position": "belowBar",
                "color": "#38bdf8",
                "shape": "circle",
                "text": "SL",
            })
        if _boolean(row.get("bos_bullish")):
            markers.append({
                "time": candle_time,
                "position": "belowBar",
                "color": "#089981",
                "shape": "arrowUp",
                "text": "BOS+",
            })
        if _boolean(row.get("bos_bearish")):
            markers.append({
                "time": candle_time,
                "position": "aboveBar",
                "color": "#f23645",
                "shape": "arrowDown",
                "text": "BOS-",
            })

    markers.sort(key=lambda marker: marker["time"])

    latest = frame.iloc[-1]
    return {
        "timeframe": timeframe,
        "label": TIMEFRAME_LABELS[timeframe],
        "candles": candles,
        "ema": ema,
        "volume": volume,
        "markers": markers,
        "levels": {
            "support": _number(latest.get("support")),
            "resistance": _number(latest.get("resistance")),
        },
        "metrics": {
            "close": _number(latest.get("close")),
            "ema_trend": str(latest.get("ema_trend", "N/A")),
            "market_structure": str(latest.get("market_structure", "N/A")),
            "rsi": _number(latest.get("rsi_14")),
            "adx": _number(latest.get("adx_14")),
            "relative_volume": _number(latest.get("relative_volume")),
        },
    }


def _build_payload(results, analysis):
    missing = {"4H", "15M", "5M"}.difference(results)
    if missing:
        raise ValueError(
            f"Missing timeframe data for TradingView chart: {sorted(missing)}"
        )

    return {
        "symbol": "BTC-USDT",
        "source": "OKX",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "analysis": analysis,
        "timeframes": [
            _prepare_timeframe(timeframe, results[timeframe])
            for timeframe in ("4H", "15M", "5M")
        ],
    }


def _build_html(payload):
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC-USDT Multi-Timeframe Technical Analysis</title>
<style>
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #070d18; color: #e2e8f0; font-family: Arial, sans-serif; }
body { width: 1800px; padding: 28px; }
.header { border: 1px solid #243247; border-radius: 14px; padding: 22px 26px; margin-bottom: 18px; background: #0d1727; }
.header-top { display: flex; justify-content: space-between; align-items: center; gap: 20px; }
h1 { margin: 0; font-size: 30px; letter-spacing: .3px; }
.source { color: #94a3b8; font-size: 15px; margin-top: 8px; }
.pills { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.pill { border: 1px solid #334155; background: #111e30; border-radius: 8px; padding: 9px 13px; font-size: 15px; }
.pill strong { color: #f8fafc; margin-left: 5px; }
.reasoning { margin-top: 16px; padding: 13px 16px; background: #111e30; border-right: 3px solid #64748b; border-radius: 7px; direction: rtl; text-align: right; line-height: 1.8; font-size: 17px; color: #dbeafe; min-height: 48px; }
.panel { border: 1px solid #243247; border-radius: 12px; overflow: hidden; margin-bottom: 16px; background: #0b1422; }
.panel-head { padding: 13px 18px; display: flex; justify-content: space-between; align-items: center; gap: 20px; background: #101c2d; border-bottom: 1px solid #243247; }
.panel-title { font-size: 20px; font-weight: 700; }
.panel-subtitle { color: #94a3b8; font-size: 14px; margin-top: 4px; }
.metrics { color: #cbd5e1; font-size: 14px; text-align: right; line-height: 1.7; }
.chart { width: 100%; height: 430px; }
.footer { color: #64748b; font-size: 12px; text-align: right; padding: 3px 4px 0; }
.footer a { color: #94a3b8; text-decoration: none; }
</style>
</head>
<body>
<div class="header">
  <div class="header-top">
    <div>
      <h1>BTC-USDT | Multi-Timeframe Technical Analysis</h1>
      <div class="source">OKX confirmed candles · TradingView Lightweight Charts™ · UTC · SH: Swing High · SL: Swing Low · BOS: Break of Structure</div>
    </div>
    <div class="pills">
      <div class="pill">AI Decision <strong id="decision">PRE-ANALYSIS</strong></div>
      <div class="pill">Market Bias <strong id="bias">—</strong></div>
      <div class="pill">Confidence <strong id="confidence">—</strong></div>
    </div>
  </div>
  <div class="reasoning" id="reasoning">Multi-timeframe market evidence is being prepared for AI analysis.</div>
</div>
<div id="charts"></div>
<div class="footer">Candles and indicators are calculated from the project's confirmed OKX data. Charting library: <a href="https://www.tradingview.com/" target="_blank">TradingView</a>.</div>
<script src="https://unpkg.com/lightweight-charts@5.3.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function () {
  const payload = __PAYLOAD__;
  const analysis = payload.analysis || null;
  const root = document.getElementById("charts");

  if (analysis) {
    document.getElementById("decision").textContent = analysis.decision || "—";
    document.getElementById("bias").textContent = analysis.market_bias || "—";
    document.getElementById("confidence").textContent =
      Number.isFinite(Number(analysis.confidence)) ? Number(analysis.confidence) + "%" : "—";
    document.getElementById("reasoning").textContent =
      analysis.reasoning || "AI analysis summary unavailable.";
    const decision = String(analysis.decision || "");
    if (decision === "WAIT") {
      document.getElementById("decision").textContent = "WAIT · NO VALIDATED ENTRY";
    }
    document.getElementById("decision").style.color =
      decision === "LONG" ? "#089981" : decision === "SHORT" ? "#f23645" : "#fbbf24";
  }

  function formatPrice(value) {
    return value == null ? "—" : Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });
  }

  function addPriceLine(series, price, color, title, style) {
    if (Number.isFinite(price) && price > 0) {
      series.createPriceLine({
        price: price,
        color: color,
        lineWidth: 1,
        lineStyle: style,
        axisLabelVisible: true,
        title: title
      });
    }
  }

  function createPanel(item) {
    const panel = document.createElement("section");
    panel.className = "panel";
    const head = document.createElement("div");
    head.className = "panel-head";
    const titleWrap = document.createElement("div");
    const title = document.createElement("div");
    title.className = "panel-title";
    title.textContent = item.timeframe + " | " + item.label;
    const subtitle = document.createElement("div");
    subtitle.className = "panel-subtitle";
    subtitle.textContent =
      "Close " + formatPrice(item.metrics.close) +
      " · EMA trend " + item.metrics.ema_trend +
      " · Structure " + item.metrics.market_structure;
    titleWrap.appendChild(title);
    titleWrap.appendChild(subtitle);

    const metrics = document.createElement("div");
    metrics.className = "metrics";
    metrics.textContent =
      "RSI " + formatPrice(item.metrics.rsi) +
      " · ADX " + formatPrice(item.metrics.adx) +
      " · Relative volume " + formatPrice(item.metrics.relative_volume);
    head.appendChild(titleWrap);
    head.appendChild(metrics);

    const chartContainer = document.createElement("div");
    chartContainer.className = "chart";
    panel.appendChild(head);
    panel.appendChild(chartContainer);
    root.appendChild(panel);

    const chart = LightweightCharts.createChart(chartContainer, {
      autoSize: true,
      height: 430,
      layout: {
        background: { type: "solid", color: "#0b1422" },
        textColor: "#cbd5e1",
        fontFamily: "Arial, sans-serif",
        attributionLogo: true
      },
      grid: {
        vertLines: { color: "#1b2a3d" },
        horzLines: { color: "#1b2a3d" }
      },
      rightPriceScale: {
        borderColor: "#334155",
        scaleMargins: { top: 0.08, bottom: 0.22 }
      },
      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
        secondsVisible: false
      },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      handleScroll: false,
      handleScale: false
    });

    const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: "#089981",
      downColor: "#f23645",
      borderUpColor: "#089981",
      borderDownColor: "#f23645",
      wickUpColor: "#089981",
      wickDownColor: "#f23645",
      lastValueVisible: true,
      priceLineVisible: true
    });
    candles.setData(item.candles);

    const emaStyles = [
      ["ema_20", "#f59e0b", "EMA 20"],
      ["ema_50", "#38bdf8", "EMA 50"],
      ["ema_200", "#c084fc", "EMA 200"]
    ];
    emaStyles.forEach(function (style) {
      const points = item.ema[style[0]];
      if (points && points.length) {
        const line = chart.addSeries(LightweightCharts.LineSeries, {
          color: style[1],
          lineWidth: 2,
          title: style[2],
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false
        });
        line.setData(points);
      }
    });

    const volume = chart.addSeries(LightweightCharts.HistogramSeries, {
      priceScaleId: "volume",
      priceFormat: { type: "volume" },
      lastValueVisible: false,
      priceLineVisible: false
    });
    volume.setData(item.volume);
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.84, bottom: 0 },
      visible: false
    });

    addPriceLine(candles, item.levels.support, "#38bdf8", "Support", 2);
    addPriceLine(candles, item.levels.resistance, "#f59e0b", "Resistance", 2);

    if (analysis && (analysis.decision === "LONG" || analysis.decision === "SHORT")) {
      const entry = analysis.entry || {};
      addPriceLine(candles, Number(entry.entry_price), "#60a5fa", "AI Entry", 0);
      addPriceLine(candles, Number(analysis.stop_loss), "#f23645", "Stop Loss", 2);
      addPriceLine(candles, Number(analysis.take_profit_1), "#089981", "TP1", 2);
      addPriceLine(candles, Number(analysis.take_profit_2), "#089981", "TP2", 2);
      addPriceLine(candles, Number(analysis.take_profit_3), "#089981", "TP3", 2);
    }

    if (item.markers.length) {
      LightweightCharts.createSeriesMarkers(candles, item.markers);
    }

    chart.timeScale().fitContent();
  }

  if (!window.LightweightCharts) {
    throw new Error("TradingView Lightweight Charts failed to load.");
  }

  payload.timeframes.forEach(createPanel);
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      window.__CHART_READY__ = true;
    });
  });
})();
</script>
</body>
</html>"""
    return template.replace("__PAYLOAD__", payload_json)


def render_tradingview_chart(results, output_path, analysis=None):
    """Render 4H, 15M and 5M OKX candles with validated chart annotations."""

    payload = _build_payload(results, analysis)
    html = _build_html(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp" + output.suffix)

    page_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            page = browser.new_page(
                viewport={"width": 1856, "height": 1800},
                device_scale_factor=1,
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.set_content(html, wait_until="load", timeout=60000)
            page.wait_for_function(
                "window.__CHART_READY__ === true",
                timeout=60000,
            )
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(400)
            page.screenshot(
                path=str(temporary_output),
                full_page=True,
                animations="disabled",
            )
        except Exception as error:
            details = "; ".join(page_errors)
            suffix = f" Browser errors: {details}" if details else ""
            raise RuntimeError(
                f"TradingView chart rendering failed: {error}.{suffix}"
            ) from error
        finally:
            browser.close()

    if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
        raise RuntimeError("TradingView chart renderer produced an empty image.")

    temporary_output.replace(output)
    return output
