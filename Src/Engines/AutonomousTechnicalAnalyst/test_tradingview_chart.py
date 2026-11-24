"""Smoke test for the TradingView multi-timeframe chart renderer."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .tradingview_chart import render_tradingview_chart


TIMEFRAME_INTERVALS_MS = {
    "4H": 4 * 60 * 60 * 1000,
    "15M": 15 * 60 * 1000,
    "5M": 5 * 60 * 1000,
}


def _build_frame(interval_ms, count=260):
    index = np.arange(count, dtype=float)
    close = 60000 + index * 9 + np.sin(index / 5) * 220
    open_price = close - np.cos(index / 4) * 35
    high = np.maximum(open_price, close) + 45
    low = np.minimum(open_price, close) - 45
    volume = 100 + (index % 19) * 7

    frame = pd.DataFrame({
        "timestamp": (
            1760000000000 + index.astype(np.int64) * interval_ms
        ),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    for period in (20, 50, 200):
        frame[f"ema_{period}"] = frame["close"].ewm(
            span=period,
            adjust=False,
        ).mean()

    frame["rsi_14"] = 50 + np.sin(index / 8) * 12
    frame["adx_14"] = 20 + np.cos(index / 11) * 5
    frame["relative_volume"] = 0.8 + (index % 7) * 0.1
    frame["swing_high"] = np.nan
    frame["swing_low"] = np.nan
    frame["bos_bullish"] = False
    frame["bos_bearish"] = False

    for position in range(12, count - 8, 24):
        frame.loc[position, "swing_high"] = high[position - 3]
        frame.loc[position + 8, "swing_low"] = low[position + 5]

    frame.loc[80, "bos_bullish"] = True
    frame.loc[170, "bos_bearish"] = True
    frame["market_structure"] = "NEUTRAL"
    frame.loc[80, "market_structure"] = "BULLISH_BOS"
    frame.loc[170, "market_structure"] = "BEARISH_BOS"
    frame["ema_trend"] = "BULLISH"
    frame["support"] = float(low[-30])
    frame["resistance"] = float(high[-20])

    return frame


def main():
    results = {
        timeframe: {
            "data": _build_frame(interval_ms),
        }
        for timeframe, interval_ms in TIMEFRAME_INTERVALS_MS.items()
    }

    wait_analysis = {
        "decision": "WAIT",
        "market_bias": "NEUTRAL",
        "confidence": 58,
        "reasoning": "آزمون نمایش چندتایم‌فریمی و نبود نقطه ورود تأییدشده.",
        "entry": {
            "entry_type": "NONE",
            "entry_price": None,
            "entry_zone_low": None,
            "entry_zone_high": None,
        },
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "take_profit_3": None,
    }

    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        pre_analysis_chart = output_dir / "pre_analysis.png"
        final_chart = output_dir / "final_wait_report.png"

        render_tradingview_chart(
            results,
            pre_analysis_chart,
        )
        render_tradingview_chart(
            results,
            final_chart,
            analysis=wait_analysis,
        )

        for chart_path in (pre_analysis_chart, final_chart):
            if not chart_path.is_file() or chart_path.stat().st_size < 10000:
                raise RuntimeError(
                    f"Chart image was not generated correctly: {chart_path.name}"
                )

    print("TRADINGVIEW CHART RENDERER TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
