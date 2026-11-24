import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

from Src.Engines.AutonomousTechnicalAnalyst import btc_scenario_monitor as monitor
from Src.Engines.AutonomousTechnicalAnalyst import btc_scenario_performance as performance


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


class TestStatusChart(unittest.TestCase):
    def test_chart_renders_forward_scenario_paths_and_target_zones(self):
        session = {
            "breakout_levels": {"LONG": 64600.0, "SHORT": 55134.0},
            "forecast": {
                "4H": {
                    "bullish": {
                        "target_zone_low": 61200.0,
                        "target_zone_high": 62600.0,
                        "invalidation_price": 59455.0,
                        "horizon_candles": 6,
                        "confidence": 35,
                    },
                    "bearish": {
                        "target_zone_low": 58000.0,
                        "target_zone_high": 59023.0,
                        "invalidation_price": 61683.9,
                        "horizon_candles": 6,
                        "confidence": 55,
                    },
                }
            },
        }
        image = monitor.build_status_chart(candles(300, 48), session)
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(image), 10000)


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


class TestReportBreakoutLevels(unittest.TestCase):
    def test_reads_same_frozen_levels_as_daily_report(self):
        report = {
            "forecast": {
                "timeframes": {
                    "4H": {
                        "bullish": {"breakout_level": 64600.0},
                        "bearish": {"breakout_level": 55134.0},
                    }
                }
            }
        }
        self.assertEqual(
            monitor.report_breakout_levels(report),
            {"LONG": 64600.0, "SHORT": 55134.0},
        )

    def test_rejects_missing_levels_instead_of_recalculating(self):
        with self.assertRaisesRegex(ValueError, "missing frozen 4H breakout levels"):
            monitor.report_breakout_levels({"forecast": {"timeframes": {"4H": {}}}})

    def test_rejects_invalid_level_order(self):
        report = {
            "forecast": {
                "timeframes": {
                    "4H": {
                        "bullish": {"breakout_level": 55000.0},
                        "bearish": {"breakout_level": 56000.0},
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "invalid ordering"):
            monitor.report_breakout_levels(report)


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


class TestTelegramStatusControls(unittest.TestCase):
    def test_periodic_status_is_disabled_by_default(self):
        state = {}
        session = {"status": "ACTIVE", "latest_checks": {}}
        with patch.object(monitor, "send_telegram") as sender:
            monitor.send_periodic_status(session, state)
        sender.assert_not_called()

    def test_status_command_can_enable_updates_for_configured_chat(self):
        state = {}
        updates = [{
            "update_id": 101,
            "message": {
                "chat": {"id": 12345},
                "text": "/status_on",
            },
        }]
        with patch.dict("os.environ", {"TELEGRAM_CHAT_ID": "12345"}), \
             patch.object(monitor, "telegram_get_updates", return_value=updates), \
             patch.object(monitor, "send_telegram", return_value=True):
            monitor.poll_telegram_commands(state)
        self.assertTrue(state["status_updates_enabled"])
        self.assertEqual(state["telegram_update_offset"], 102)

    def test_periodic_status_is_sent_once_per_new_closed_candle(self):
        state = {"status_updates_enabled": True}
        session = {
            "status": "ACTIVE",
            "analysis_id": "test-report",
            "scenarios": {
                "LONG": {"status": "WAITING"},
                "SHORT": {"status": "WAITING"},
            },
            "latest_checks": {
                "LONG": {"observed_timestamp": 1800, "observed_price": 60000.0,
                         "checks": {"4H_BREAKOUT": {"known": True, "met": False},
                                    "15M_MOMENTUM": {"known": True, "met": True},
                                    "5M_MOMENTUM": {"known": True, "met": True}}},
                "SHORT": {"observed_timestamp": 1800, "observed_price": 60000.0,
                          "checks": {"4H_BREAKOUT": {"known": True, "met": False},
                                     "15M_MOMENTUM": {"known": True, "met": True},
                                     "5M_MOMENTUM": {"known": True, "met": True}}},
            },
        }
        with patch.object(monitor, "send_telegram", return_value=True) as sender:
            monitor.send_periodic_status(session, state)
            monitor.send_periodic_status(session, state)
        sender.assert_called_once()
        self.assertEqual(state["status_update_last_candle"], 1800)

    def test_periodic_status_uses_chart_photo_when_confirmed_candles_exist(self):
        state = {"status_updates_enabled": True}
        session = {
            "status": "ACTIVE",
            "analysis_id": "test-report",
            "breakout_levels": {"LONG": 65000.0, "SHORT": 62000.0},
            "scenarios": {
                "LONG": {"status": "WAITING"},
                "SHORT": {"status": "WAITING"},
            },
            "latest_checks": {
                "LONG": {"observed_timestamp": 1800, "observed_price": 64000.0,
                         "checks": {"4H_BREAKOUT": {"known": True, "met": False}}},
                "SHORT": {"observed_timestamp": 1800, "observed_price": 64000.0,
                          "checks": {"4H_BREAKOUT": {"known": True, "met": False}}},
            },
        }
        five_minute = candles(300, 5, start=600, mode="up")
        with patch.object(monitor, "build_status_chart", return_value=b"png") as chart, \
             patch.object(monitor, "send_telegram_photo", return_value=True) as photo, \
             patch.object(monitor, "send_telegram") as text_sender:
            monitor.send_periodic_status(session, state, {"5M": five_minute})
        chart.assert_called_once_with(five_minute, session)
        photo.assert_called_once()
        text_sender.assert_not_called()
        self.assertEqual(state["status_update_last_candle"], 1800)

    def test_finalized_session_does_not_send_periodic_status(self):
        state = {"status_updates_enabled": True}
        session = {"status": "NO_SCENARIO_ACTIVATED", "latest_checks": {}}
        with patch.object(monitor, "send_telegram") as sender:
            monitor.send_periodic_status(session, state)
        sender.assert_not_called()


class TestScenarioPerformance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.observations_path = self.root / "observations.json"
        self.outcomes_path = self.root / "outcomes.json"
        self.report_path = self.root / "performance.json"
        self.entry_time = 1_700_000_000

    def tearDown(self):
        self.temp_dir.cleanup()

    def active_session(self):
        return {
            "analysis_id": "performance-test",
            "status": "ACTIVE",
            "scenarios": {
                "LONG": {"status": "WAITING", "activation": None},
                "SHORT": {"status": "WAITING", "activation": None},
            },
            "latest_checks": {
                "LONG": {
                    "observed_timestamp": self.entry_time,
                    "observed_price": 100.0,
                    "checks": {
                        "4H_BREAKOUT": {"known": True, "met": True, "level": 99.0},
                        "15M_MOMENTUM": {"known": True, "met": True},
                        "5M_MOMENTUM": {"known": True, "met": False},
                    },
                },
                "SHORT": {
                    "observed_timestamp": self.entry_time,
                    "observed_price": 100.0,
                    "checks": {
                        "4H_BREAKOUT": {"known": True, "met": False, "level": 90.0},
                        "15M_MOMENTUM": {"known": True, "met": False},
                        "5M_MOMENTUM": {"known": True, "met": False},
                    },
                },
            },
        }

    def test_observations_are_deduplicated_and_near_miss_is_labeled(self):
        session = self.active_session()
        self.assertTrue(performance.record_observation(session, self.observations_path))
        self.assertFalse(performance.record_observation(session, self.observations_path))
        store = performance.read_store(self.observations_path, {})
        self.assertEqual(len(store["records"]), 1)
        self.assertEqual(
            store["records"][0]["scenarios"]["LONG"]["classification"],
            "UNCONFIRMED_BREAKOUT_NEAR_MISS",
        )

    def test_activation_outcomes_resolve_all_forward_horizons(self):
        session = self.active_session()
        session["scenarios"]["LONG"] = {
            "status": "ACTIVATED",
            "activation": {
                "timestamp_unix": self.entry_time,
                "price": 100.0,
                "checks": {"4H_BREAKOUT": {"level": 99.0}},
            },
        }
        five_minute = []
        for index in range(288):
            close = 100.0 + index * 0.01
            five_minute.append({
                "timestamp": self.entry_time + index * 300,
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1.0,
            })
        four_hour = [{
            "timestamp": self.entry_time,
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1.0,
        }]
        store = performance.update_outcomes(
            session,
            {"5M": five_minute, "4H": four_hour},
            self.outcomes_path,
            now_timestamp=self.entry_time + 86400,
        )
        activated = [item for item in store["records"] if item["kind"] == "ACTIVATED"]
        self.assertEqual(len(activated), 1)
        self.assertTrue(all(item["resolved"] for item in activated[0]["windows"].values()))
        self.assertEqual(activated[0]["breakout_assessment"]["status"], "HELD_THROUGH_4H")

        report = performance.build_performance_report(
            {"records": []}, store, generated_at=self.entry_time + 86400
        )
        self.assertEqual(report["scenarios"]["LONG"]["activated_count"], 1)
        self.assertEqual(report["scenarios"]["LONG"]["horizons"]["24H"]["resolved_count"], 1)

    def test_near_miss_candidate_is_not_counted_as_activation(self):
        session = self.active_session()
        store = performance.update_outcomes(
            session,
            {"5M": [], "4H": []},
            self.outcomes_path,
            now_timestamp=self.entry_time,
        )
        self.assertEqual(len(store["records"]), 1)
        self.assertEqual(store["records"][0]["kind"], "UNCONFIRMED_NEAR_MISS")
        report = performance.build_performance_report({"records": []}, store)
        self.assertEqual(report["scenarios"]["LONG"]["activated_count"], 0)
        self.assertEqual(report["scenarios"]["LONG"]["near_miss_candidate_count"], 1)
        self.assertEqual(report["scenarios"]["LONG"]["near_miss_horizons"]["4H"]["resolved_count"], 0)


    def test_direction_report_includes_condition_rates_and_comparison(self):
        observations = {"records": []}
        for index in range(4):
            observations["records"].append({
                "record_id": f"sample:{index}",
                "scenarios": {
                    "LONG": {
                        "status": "WAITING",
                        "classification": (
                            "MOMENTUM_READY_BREAKOUT_PENDING" if index == 0 else "WAITING"
                        ),
                        "near_miss": False,
                        "checks": {
                            "4H_BREAKOUT": {"known": True, "met": index == 0},
                            "15M_MOMENTUM": {"known": True, "met": True},
                            "5M_MOMENTUM": {"known": index != 3, "met": index != 3},
                        },
                    },
                    "SHORT": {
                        "status": "WAITING",
                        "classification": (
                            "UNCONFIRMED_BREAKOUT_NEAR_MISS" if index == 1 else "WAITING"
                        ),
                        "near_miss": index == 1,
                        "checks": {
                            "4H_BREAKOUT": {"known": True, "met": index == 1},
                            "15M_MOMENTUM": {"known": True, "met": index != 1},
                            "5M_MOMENTUM": {"known": True, "met": index != 1},
                        },
                    },
                },
            })
        outcomes = {"records": []}
        report = performance.build_performance_report(
            observations, outcomes, generated_at=self.entry_time
        )
        long_report = report["scenarios"]["LONG"]
        short_report = report["scenarios"]["SHORT"]
        self.assertEqual(long_report["monitoring_observations"], 4)
        self.assertEqual(
            long_report["condition_statistics"]["15M_MOMENTUM"]["pass_rate_pct"],
            100.0,
        )
        self.assertEqual(
            long_report["condition_statistics"]["5M_MOMENTUM"]["known_count"],
            3,
        )
        self.assertEqual(
            long_report["momentum_ready_breakout_pending_observation_count"],
            1,
        )
        self.assertEqual(
            short_report["near_miss_observation_count"],
            1,
        )
        self.assertEqual(
            report["direction_comparison"]["by_horizon"]["4H"][
                "higher_average_forward_return_direction"
            ],
            "INSUFFICIENT_DATA",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
