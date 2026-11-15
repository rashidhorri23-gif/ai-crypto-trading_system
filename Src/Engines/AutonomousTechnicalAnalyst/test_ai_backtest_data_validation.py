import json
import unittest

import pandas as pd

from Src.Engines.AutonomousTechnicalAnalyst.ai_backtest import (
    validate_backtest_data,
    build_prompt,
)


def make_market_data(count=3):
    rows = []

    start_time = pd.Timestamp(
        "2026-01-01 00:00:00",
        tz="UTC",
    )

    for index in range(count):
        candle_time = start_time + pd.Timedelta(hours=4 * index)

        rows.append({
            "timestamp": int(candle_time.timestamp() * 1000),
            "timestamp_utc": candle_time,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        })

    return pd.DataFrame(rows)


class TestBacktestDataValidation(unittest.TestCase):

    def test_valid_market_data(self):
        data = make_market_data()

        self.assertTrue(validate_backtest_data(data))

    def test_duplicate_timestamps_are_rejected(self):
        data = make_market_data()
        data.loc[2, "timestamp"] = data.loc[1, "timestamp"]

        with self.assertRaises(ValueError):
            validate_backtest_data(data)

    def test_irregular_candle_intervals_are_rejected(self):
        data = make_market_data()
        data.loc[2, "timestamp"] += 3600000

        with self.assertRaises(ValueError):
            validate_backtest_data(data)

    def test_invalid_ohlc_geometry_is_rejected(self):
        data = make_market_data()
        data.loc[1, "high"] = 90.0

        with self.assertRaises(ValueError):
            validate_backtest_data(data)

    def test_missing_required_column_is_rejected(self):
        data = make_market_data()
        data = data.drop(columns=["volume"])

        with self.assertRaises(ValueError):
            validate_backtest_data(data)

    def test_empty_data_is_rejected(self):
        data = make_market_data()
        data = data.iloc[0:0]

        with self.assertRaises(ValueError):
            validate_backtest_data(data)


class TestDecisionTimestamp(unittest.TestCase):

    def test_decision_time_is_candle_close_time(self):
        data = make_market_data()

        prompt = build_prompt(data, 0)

        expected_time = "2026-01-01T04:00:00+00:00"

        self.assertIn(
            f'"decision_time_utc": "{expected_time}"',
            prompt,
        )

        def test_prompt_contains_only_available_history(self):
        data = make_market_data(count=3)
        prompt = build_prompt(data, 1)

        marker = "Historical market data:\n\n"
        prompt_data = json.loads(
            prompt.split(marker, 1)[1]
        )

        self.assertEqual(
            prompt_data["decision_time_utc"],
            "2026-01-01T08:00:00+00:00",
        )

        history_times = [
            item["timestamp_utc"]
            for item in prompt_data["recent_market_history"]
        ]

        self.assertEqual(
            history_times,
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T04:00:00+00:00",
            ],
        )

        self.assertNotIn(
            "2026-01-01T08:00:00+00:00",
            history_times,
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
