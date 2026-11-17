
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Src.Engines.AutonomousTechnicalAnalyst import monitor_btc_signal as monitor


VALID_LONG_PLAN = {
    "setup_type": "RETEST",
    "entry_zone_low": 100.0,
    "entry_zone_high": 105.0,
    "stop_loss": 95.0,
    "take_profit_1": 110.0,
    "take_profit_2": 115.0,
    "take_profit_3": 120.0,
}

VALID_SHORT_PLAN = {
    "setup_type": "BREAKDOWN",
    "entry_zone_low": 95.0,
    "entry_zone_high": 100.0,
    "stop_loss": 105.0,
    "take_profit_1": 90.0,
    "take_profit_2": 85.0,
    "take_profit_3": 80.0,
}


def make_candle(timestamp, open_price, high, low, close):
    return {
        "timestamp": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100.0,
        "confirm": "1",
    }


class TestPlanValidation(unittest.TestCase):

    def test_valid_long_plan(self):
        self.assertTrue(
            monitor.validate_plan("LONG", VALID_LONG_PLAN)
        )

    def test_valid_short_plan(self):
        self.assertTrue(
            monitor.validate_plan("SHORT", VALID_SHORT_PLAN)
        )

    def test_invalid_long_stop_loss(self):
        plan = dict(VALID_LONG_PLAN)
        plan["stop_loss"] = 102.0

        with self.assertRaises(ValueError):
            monitor.validate_plan("LONG", plan)

    def test_invalid_short_targets(self):
        plan = dict(VALID_SHORT_PLAN)
        plan["take_profit_1"] = 110.0

        with self.assertRaises(ValueError):
            monitor.validate_plan("SHORT", plan)

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            monitor.validate_plan("BUY", VALID_LONG_PLAN)


class TestBreakoutDetection(unittest.TestCase):

    def test_long_breakout_after_baseline(self):
        scenario = {
            "baseline_4h_timestamp": 0,
            "breakout_timestamp": None,
        }

        candles = [
            make_candle(0, 98, 101, 97, 99),
            make_candle(14400, 99, 108, 98, 106),
        ]

        result = monitor.find_breakout(
            "LONG",
            VALID_LONG_PLAN,
            scenario,
            candles,
        )

        self.assertEqual(result, 28800)

    def test_short_breakout_after_baseline(self):
        scenario = {
            "baseline_4h_timestamp": 0,
            "breakout_timestamp": None,
        }

        candles = [
            make_candle(0, 102, 103, 99, 101),
            make_candle(14400, 101, 102, 92, 94),
        ]

        result = monitor.find_breakout(
            "SHORT",
            VALID_SHORT_PLAN,
            scenario,
            candles,
        )

        self.assertEqual(result, 28800)

    def test_old_candle_cannot_trigger_breakout(self):
        scenario = {
            "baseline_4h_timestamp": 14400,
            "breakout_timestamp": None,
        }

        candles = [
            make_candle(0, 99, 108, 98, 106),
            make_candle(14400, 99, 104, 98, 103),
        ]

        result = monitor.find_breakout(
            "LONG",
            VALID_LONG_PLAN,
            scenario,
            candles,
        )

        self.assertIsNone(result)


class TestRetestDetection(unittest.TestCase):

    def test_long_retest_confirmation(self):
        candle = make_candle(28800, 102, 106, 99, 104)

        result = monitor.find_retest(
            "LONG",
            VALID_LONG_PLAN,
            28800,
            [candle],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["close"], 104)

    def test_short_retest_confirmation(self):
        candle = make_candle(28800, 99, 101, 94, 96)

        result = monitor.find_retest(
            "SHORT",
            VALID_SHORT_PLAN,
            28800,
            [candle],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["close"], 96)

    def test_retest_before_breakout_close_is_rejected(self):
        candle = make_candle(28500, 102, 106, 99, 104)

        result = monitor.find_retest(
            "LONG",
            VALID_LONG_PLAN,
            28800,
            [candle],
        )

        self.assertIsNone(result)

    def test_wrong_candle_direction_is_rejected(self):
        candle = make_candle(28800, 104, 106, 99, 102)

        result = monitor.find_retest(
            "LONG",
            VALID_LONG_PLAN,
            28800,
            [candle],
        )

        self.assertIsNone(result)


class TestScenarioState(unittest.TestCase):

    def test_same_fingerprint_keeps_existing_scenario(self):
        fingerprint = monitor.make_fingerprint(
            "LONG",
            VALID_LONG_PLAN,
        )

        existing = {
            "fingerprint": fingerprint,
            "baseline_4h_timestamp": 100,
            "breakout_timestamp": 200,
            "pending_retest": None,
            "alert_sent": False,
            "invalidated": False,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

        state = {
            "scenarios": {"LONG": existing},
            "sent_fingerprints": [],
        }

        latest_4h = make_candle(500, 100, 106, 99, 104)

        result = monitor.initialize_scenario(
            "LONG",
            VALID_LONG_PLAN,
            latest_4h,
            state,
        )

        self.assertIs(result, existing)
        self.assertEqual(result["baseline_4h_timestamp"], 100)

    def test_previously_sent_fingerprint_is_not_reactivated(self):
        fingerprint = monitor.make_fingerprint(
            "LONG",
            VALID_LONG_PLAN,
        )

        state = {
            "scenarios": {},
            "sent_fingerprints": [fingerprint],
        }

        latest_4h = make_candle(500, 100, 106, 99, 104)

        result = monitor.initialize_scenario(
            "LONG",
            VALID_LONG_PLAN,
            latest_4h,
            state,
        )

        self.assertTrue(result["alert_sent"])


class TestInvalidation(unittest.TestCase):

    def test_long_scenario_invalidation(self):
        scenario = {"breakout_timestamp": 28800}

        latest_4h = make_candle(28800, 96, 98, 90, 93)

        result = monitor.check_invalidation(
            "LONG",
            VALID_LONG_PLAN,
            scenario,
            latest_4h,
        )

        self.assertTrue(result)

    def test_short_scenario_invalidation(self):
        scenario = {"breakout_timestamp": 28800}

        latest_4h = make_candle(28800, 104, 110, 103, 107)

        result = monitor.check_invalidation(
            "SHORT",
            VALID_SHORT_PLAN,
            scenario,
            latest_4h,
        )

        self.assertTrue(result)


class TestDuplicateAlertPrevention(unittest.TestCase):

    def test_alert_is_sent_only_once(self):
        fingerprint = monitor.make_fingerprint(
            "LONG",
            VALID_LONG_PLAN,
        )

        state = {
            "version": 1,
            "scenarios": {
                "LONG": {
                    "fingerprint": fingerprint,
                    "baseline_4h_timestamp": 0,
                    "breakout_timestamp": None,
                    "pending_retest": None,
                    "alert_sent": False,
                    "invalidated": False,
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            },
            "sent_fingerprints": [],
            "last_processed_5m_timestamp": None,
        }

        candles_4h = [
            make_candle(0, 98, 101, 97, 99),
            make_candle(14400, 99, 108, 98, 106),
        ]

        candles_5m = [
            make_candle(28800, 102, 106, 99, 104),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            state_file = Path(temp_dir) / "monitor_state.json"

            with patch.object(monitor, "STATE_FILE", state_file):
                with patch.object(
                    monitor,
                    "send_telegram",
                    return_value=True,
                ) as telegram_mock:

                    monitor.process_scenario(
                        "LONG",
                        VALID_LONG_PLAN,
                        state,
                        candles_4h,
                        candles_5m,
                    )

                    self.assertTrue(
                        state["scenarios"]["LONG"]["alert_sent"]
                    )

                    self.assertEqual(telegram_mock.call_count, 1)

                    monitor.process_scenario(
                        "LONG",
                        VALID_LONG_PLAN,
                        state,
                        candles_4h,
                        candles_5m,
                    )

                    self.assertEqual(telegram_mock.call_count, 1)


class TestCheckpointProtection(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temp_dir.name) / "state.json"

        self.initial_checkpoint = 1000
        self.latest_checkpoint = 2000

        self.state = {
            "version": 1,
            "scenarios": {},
            "sent_fingerprints": [],
            "last_processed_5m_timestamp": self.initial_checkpoint,
        }

        self.signal = {
            "bullish_plan": VALID_LONG_PLAN,
            "bearish_plan": VALID_SHORT_PLAN,
        }

        self.candles_4h = [
            make_candle(0, 98, 101, 97, 99),
            make_candle(14400, 99, 104, 98, 103),
        ]

        self.candles_5m = [
            make_candle(1500, 101, 103, 99, 102),
            make_candle(2000, 102, 104, 100, 103),
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_main_with_failure(self, failed_direction):
        def fake_process(
            direction,
            plan,
            state,
            candles_4h,
            new_candles_5m,
        ):
            if direction == failed_direction:
                raise RuntimeError("Simulated processing failure")

        with patch.object(monitor, "SIGNAL_FILE", Path("unused.json")):
            with patch.object(monitor, "STATE_FILE", self.state_file):
                with patch.object(
                    monitor,
                    "load_json",
                    return_value=self.signal,
                ):
                    with patch.object(
                        monitor,
                        "load_state",
                        return_value=self.state,
                    ):
                        with patch.object(
                            monitor,
                            "fetch_candles",
                            side_effect=[
                                self.candles_4h,
                                self.candles_5m,
                            ],
                        ):
                            with patch.object(
                                monitor,
                                "process_scenario",
                                side_effect=fake_process,
                            ):
                                monitor.main()

    def test_long_failure_does_not_advance_checkpoint(self):
        self.run_main_with_failure("LONG")

        self.assertEqual(
            self.state["last_processed_5m_timestamp"],
            self.initial_checkpoint,
        )

    def test_short_failure_does_not_advance_checkpoint(self):
        self.run_main_with_failure("SHORT")

        self.assertEqual(
            self.state["last_processed_5m_timestamp"],
            self.initial_checkpoint,
        )

    def test_successful_processing_advances_checkpoint(self):
        with patch.object(monitor, "SIGNAL_FILE", Path("unused.json")):
            with patch.object(monitor, "STATE_FILE", self.state_file):
                with patch.object(
                    monitor,
                    "load_json",
                    return_value=self.signal,
                ):
                    with patch.object(
                        monitor,
                        "load_state",
                        return_value=self.state,
                    ):
                        with patch.object(
                            monitor,
                            "fetch_candles",
                            side_effect=[
                                self.candles_4h,
                                self.candles_5m,
                            ],
                        ):
                            with patch.object(
                                monitor,
                                "process_scenario",
                                return_value=None,
                            ):
                                monitor.main()

        self.assertEqual(
            self.state["last_processed_5m_timestamp"],
            self.latest_checkpoint,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
