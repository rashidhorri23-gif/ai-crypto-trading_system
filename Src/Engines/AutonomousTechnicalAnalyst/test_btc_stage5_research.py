import tempfile
import unittest
from pathlib import Path

from Src.Engines.AutonomousTechnicalAnalyst import btc_stage5_research as research


class StageFiveResearchTests(unittest.TestCase):
    def test_ema_series(self):
        self.assertEqual(research.ema_series([10, 10, 10], 3), [10.0, 10.0, 10.0])

    def test_rsi_series_flat_market(self):
        values = research.rsi_series([100.0] * 20)
        self.assertEqual(values[14], 50.0)
        self.assertEqual(values[-1], 50.0)

    def test_directional_momentum_thresholds(self):
        long_snapshot = {"known": True, "trend": "BULLISH", "rsi": 56.0, "macd_histogram": 1.0}
        short_snapshot = {"known": True, "trend": "BEARISH", "rsi": 44.0, "macd_histogram": -1.0}
        self.assertTrue(research.is_aligned(long_snapshot, "LONG", 50.0))
        self.assertTrue(research.is_aligned(short_snapshot, "SHORT", 50.0))
        self.assertFalse(research.is_aligned(long_snapshot, "LONG", 55.0))
        self.assertFalse(research.is_aligned(short_snapshot, "SHORT", 55.0))

    def test_load_candles_rejects_duplicate_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candles.csv"
            path.write_text(
                "timestamp,open,high,low,close,volume\n"
                "1000,10,11,9,10,1\n1000,10,11,9,10,1\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                research.load_candles(path, "5m")

    def test_summary_handles_no_signals(self):
        result = research.summarize([])
        self.assertEqual(result["signal_count"], 0)
        self.assertIsNone(result["horizons"]["4H"]["average_forward_return_pct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
