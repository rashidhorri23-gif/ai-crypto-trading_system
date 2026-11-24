import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from Src.Engines.AutonomousTechnicalAnalyst import btc_scenario_monitor as monitor


def candles(interval, count, start=1_700_000_000, mode="up"):
    rows = []
    for index in range(count):
        if mode == "up":
            close = 100.0 + index * 0.2
        elif mode == "down":
            close = 200.0 - index * 0.2
        else:
            close = 100.0 + (index % 4) * 0.01
        rows.append({
            "timestamp": start + index * interval,
            "open": close - 0.02,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 10.0,
        })
    return rows


class TestIndicators(unittest.TestCase):
    def test_ema_requires_full_period(self):
        self.assertIsNone(monitor.ema([1.0, 2.0], 3))
        self.assertAlmostEqual(monitor.ema([1.0, 2.0, 3.0], 3), 2.25)

    def test_rsi_rising_series_is_100(self):
        self.assertEqual(monitor.rsi([float(i) for i in range(20)]), 100.0)

    def test_momentum_snapshot_reports_trend(self):
        result = monitor.momentum_snapshot(candles(300, 240, mode="up"))
        self.assertTrue(result["known"])
        self.assertEqual(result["ema_trend"], "BULLISH")
        self.assertGreater(result["rsi"], 50)


class TestScenarioEvaluation(unittest.TestCase):
    def setUp(self):
        self.generated_at = 1_700_000_000
        self.frames = {
            "4H": candles(14400, 80, start=self.generated_at - 80 * 14400 + 60, mode="up"),
            "15M": candles(900, 240, start=self.generated_at - 240 * 900 + 60, mode="up"),
            "5M": candles(300, 240, start=self.generated_at - 240 * 300 + 60, mode="up"),
        }

    def test_long_requires_all_three_timeframes(self):
        self.frames["4H"][-1]["close"] = max(c["high"] for c in self.frames["4H"][-21:-1]) + 1
        self.frames["4H"][-1]["high"] = self.frames["4H"][-1]["close"] + 0.1
        result = monitor.evaluate_scenarios(self.frames, self.generated_at)
        self.assertTrue(result["LONG"]["ready"])
        self.assertFalse(result["SHORT"]["ready"])

    def test_short_requires_all_three_timeframes(self):
        down = {
            "4H": candles(14400, 80, start=self.generated_at - 80 * 14400 + 60, mode="down"),
            "15M": candles(900, 240, start=self.generated_at - 240 * 900 + 60, mode="down"),
            "5M": candles(300, 240, start=self.generated_at - 240 * 300 + 60, mode="down"),
        }
        down["4H"][-1]["close"] = min(c["low"] for c in down["4H"][-21:-1]) - 1
        down["4H"][-1]["low"] = down["4H"][-1]["close"] - 0.1
        result = monitor.evaluate_scenarios(down, self.generated_at)
        self.assertTrue(result["SHORT"]["ready"])
        self.assertFalse(result["LONG"]["ready"])

    def test_momentum_failure_blocks_activation(self):
        self.frames["4H"][-1]["close"] = max(c["high"] for c in self.frames["4H"][-21:-1]) + 1
        self.frames["4H"][-1]["high"] = self.frames["4H"][-1]["close"] + 0.1
        self.frames["5M"] = candles(300, 240, start=self.generated_at - 240 * 300 + 60, mode="down")
        result = monitor.evaluate_scenarios(self.frames, self.generated_at)
        self.assertFalse(result["LONG"]["ready"])

    def test_pre_report_candles_cannot_activate(self):
        before_report = {
            "4H": candles(14400, 80, start=self.generated_at - 80 * 14400 - 60, mode="up"),
            "15M": candles(900, 240, start=self.generated_at - 240 * 900 - 60, mode="up"),
            "5M": candles(300, 240, start=self.generated_at - 240 * 300 - 60, mode="up"),
        }
        before_report["4H"][-1]["close"] = max(c["high"] for c in before_report["4H"][-21:-1]) + 1
        before_report["4H"][-1]["high"] = before_report["4H"][-1]["close"] + 0.1
        result = monitor.evaluate_scenarios(before_report, self.generated_at)
        self.assertFalse(result["LONG"]["ready"])


class TestSession(unittest.TestCase):
    def test_no_scenario_activated_is_recorded_at_expiry(self):
        report = {
            "analysis_id": "test-report",
            "forecast": {"timeframes": {}},
        }
        generated = datetime(2026, 11, 23, 4, 0, tzinfo=timezone.utc).timestamp()
        state = {"current_session": None, "history": [], "notification_ids": []}
        with patch.object(monitor, "send_telegram", return_value=True):
            session = monitor.process_session(
                report, generated, state,
                now_timestamp=monitor.next_baku_0800_timestamp(generated) + 1,
                candles_by_tf={},
            )
        self.assertEqual(session["status"], "NO_SCENARIO_ACTIVATED")
        self.assertEqual(state["history"][-1]["status"], "NO_SCENARIO_ACTIVATED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
