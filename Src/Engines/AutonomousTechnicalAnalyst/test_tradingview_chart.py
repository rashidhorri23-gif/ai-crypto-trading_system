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
        "forecast": {
            "timeframes": {
                "4H": {
                    "bullish": {
                        "target_zone_low": 63000.0,
                        "target_zone_high": 63500.0,
                        "invalidation_price": 59000.0,
                        "horizon_candles": 6,
                        "confidence": 62,
                        "trigger": "شکست مقاومت تایید شود."
                    },
                    "bearish": {
                        "target_zone_low": 57000.0,
                        "target_zone_high": 57500.0,
                        "invalidation_price": 64000.0,
                        "horizon_candles": 6,
                        "confidence": 68,
                        "trigger": "حمایت از دست برود."
                    }
                },
                "15M": {
                    "bullish": {
                        "target_zone_low": 62500.0,
                        "target_zone_high": 62700.0,
                        "invalidation_price": 60000.0,
                        "horizon_candles": 8,
                        "confidence": 55,
                        "trigger": "تثبیت بالای مقاومت."
                    },
                    "bearish": {
                        "target_zone_low": 59500.0,
                        "target_zone_high": 59800.0,
                        "invalidation_price": 63000.0,
                        "horizon_candles": 8,
                        "confidence": 61,
                        "trigger": "شکست حمایت کوتاه‌مدت."
                    }
                },
                "5M": {
                    "bullish": {
                        "target_zone_low": 62200.0,
                        "target_zone_high": 62300.0,
                        "invalidation_price": 61000.0,
                        "horizon_candles": 12,
                        "confidence": 51,
                        "trigger": "بازپس‌گیری سقف محلی."
                    },
                    "bearish": {
                        "target_zone_low": 60800.0,
                        "target_zone_high": 60900.0,
                        "invalidation_price": 62500.0,
                        "horizon_candles": 12,
                        "confidence": 57,
                        "trigger": "شکست کف محلی."
                    }
                }
            }
        },
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

    long_analysis = {
        "decision": "LONG",
        "market_bias": "BULLISH",
        "confidence": 71,
        "reasoning": "آزمون نمایش ورود، محدوده ورود، حد ضرر و اهداف.",
        "forecast": {
            "timeframes": {
                "4H": {
                    "bullish": {
                        "target_zone_low": 63000.0,
                        "target_zone_high": 63500.0,
                        "invalidation_price": 59000.0,
                        "horizon_candles": 6,
                        "confidence": 62,
                        "trigger": "شکست مقاومت تایید شود."
                    },
                    "bearish": {
                        "target_zone_low": 57000.0,
                        "target_zone_high": 57500.0,
                        "invalidation_price": 64000.0,
                        "horizon_candles": 6,
                        "confidence": 68,
                        "trigger": "حمایت از دست برود."
                    }
                },
                "15M": {
                    "bullish": {
                        "target_zone_low": 62500.0,
                        "target_zone_high": 62700.0,
                        "invalidation_price": 60000.0,
                        "horizon_candles": 8,
                        "confidence": 55,
                        "trigger": "تثبیت بالای مقاومت."
                    },
                    "bearish": {
                        "target_zone_low": 59500.0,
                        "target_zone_high": 59800.0,
                        "invalidation_price": 63000.0,
                        "horizon_candles": 8,
                        "confidence": 61,
                        "trigger": "شکست حمایت کوتاه‌مدت."
                    }
                },
                "5M": {
                    "bullish": {
                        "target_zone_low": 62200.0,
                        "target_zone_high": 62300.0,
                        "invalidation_price": 61000.0,
                        "horizon_candles": 12,
                        "confidence": 51,
                        "trigger": "بازپس‌گیری سقف محلی."
                    },
                    "bearish": {
                        "target_zone_low": 60800.0,
                        "target_zone_high": 60900.0,
                        "invalidation_price": 62500.0,
                        "horizon_candles": 12,
                        "confidence": 57,
                        "trigger": "شکست کف محلی."
                    }
                }
            }
        },
        "entry": {
            "entry_type": "LIMIT",
            "entry_price": 60000.0,
            "entry_zone_low": 59900.0,
            "entry_zone_high": 60100.0,
        },
        "stop_loss": 59000.0,
        "take_profit_1": 61000.0,
        "take_profit_2": 62000.0,
        "take_profit_3": 63000.0,
    }

    with tempfile.TemporaryDirectory() as directory:
        output_dir = Path(directory)
        pre_analysis_chart = output_dir / "pre_analysis.png"
        wait_chart = output_dir / "final_wait_report.png"
        long_chart = output_dir / "final_long_report.png"

        render_tradingview_chart(
            results,
            pre_analysis_chart,
        )
        render_tradingview_chart(
            results,
            wait_chart,
            analysis=wait_analysis,
        )
        render_tradingview_chart(
            results,
            long_chart,
            analysis=long_analysis,
        )

        for chart_path in (pre_analysis_chart, wait_chart, long_chart):
            if not chart_path.is_file() or chart_path.stat().st_size < 10000:
                raise RuntimeError(
                    f"Chart image was not generated correctly: {chart_path.name}"
                )

    print("TRADINGVIEW CHART RENDERER TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
