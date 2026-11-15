
import unittest

import pandas as pd

from .ai_backtest import execute_trade


def make_market_data(candles):
    rows = []

    for index, candle in enumerate(candles):
        rows.append({
            "timestamp_utc": pd.Timestamp(
                "2026-01-01 00:00:00",
                tz="UTC"
            ) + pd.Timedelta(hours=4 * index),
            "open": candle[0],
            "high": candle[1],
            "low": candle[2],
            "close": candle[3],
            "volume": 1000,
        })

    return pd.DataFrame(rows)


class TestTradeExecution(unittest.TestCase):

    def setUp(self):
        self.long_decision = {
            "decision": "LONG",
            "decision_index": 0,
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "confidence": 90,
            "reason": "Test long setup",
        }

        self.short_decision = {
            "decision": "SHORT",
            "decision_index": 0,
            "entry": 100,
            "stop_loss": 105,
            "take_profit": 90,
            "confidence": 90,
            "reason": "Test short setup",
        }

    def test_long_take_profit(self):
        data = make_market_data([
            (100, 101, 99, 100),
            (100, 111, 99, 108),
        ])

        trade = execute_trade(
            data,
            self.long_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(trade["exit_price"], 110)
        self.assertGreater(trade["net_r"], 0)

    def test_short_take_profit(self):
        data = make_market_data([
            (100, 101, 99, 100),
            (100, 101, 89, 92),
        ])

        trade = execute_trade(
            data,
            self.short_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(trade["exit_price"], 90)
        self.assertGreater(trade["net_r"], 0)

    def test_long_stop_loss(self):
        data = make_market_data([
            (100, 101, 99, 100),
            (100, 105, 94, 96),
        ])

        trade = execute_trade(
            data,
            self.long_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertEqual(trade["exit_price"], 95)
        self.assertLess(trade["net_r"], 0)

    def test_short_stop_loss(self):
        data = make_market_data([
            (100, 101, 99, 100),
            (100, 106, 95, 104),
        ])

        trade = execute_trade(
            data,
            self.short_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertEqual(trade["exit_price"], 105)
        self.assertLess(trade["net_r"], 0)

    def test_stop_first_when_both_levels_are_touched(self):
        data = make_market_data([
            (100, 101, 99, 100),
            (100, 111, 94, 105),
        ])

        trade = execute_trade(
            data,
            self.long_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertEqual(trade["exit_price"], 95)

    def test_entry_candle_stop_loss(self):
        data = make_market_data([
            (100, 101, 94, 96),
            (96, 100, 95, 98),
        ])

        trade = execute_trade(
            data,
            self.long_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(
            trade["exit_reason"],
            "STOP_LOSS_ENTRY_CANDLE",
        )
        self.assertEqual(trade["exit_index"], 0)

    def test_time_exit(self):
        data = make_market_data([
            (100, 101, 99, 100),
            (100, 104, 98, 102),
            (102, 105, 99, 103),
        ])

        trade = execute_trade(
            data,
            self.long_decision,
            0,
            100,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "TIME_EXIT")
        self.assertEqual(trade["exit_index"], 2)
        self.assertEqual(trade["exit_price"], 103)

    def test_reject_long_when_slippage_crosses_target(self):
        decision = {
            "decision": "LONG",
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 100.05,
        }

        data = make_market_data([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ])

        trade = execute_trade(
            data,
            decision,
            0,
            100,
        )

        self.assertIsNone(trade)

    def test_reject_short_when_slippage_crosses_target(self):
        decision = {
            "decision": "SHORT",
            "entry": 100,
            "stop_loss": 105,
            "take_profit": 99.95,
        }

        data = make_market_data([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ])

        trade = execute_trade(
            data,
            decision,
            0,
            100,
        )

        self.assertIsNone(trade)

    def test_reject_invalid_direction(self):
        decision = {
            "decision": "WAIT",
            "entry": 0,
            "stop_loss": 0,
            "take_profit": 0,
        }

        data = make_market_data([
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ])

        trade = execute_trade(
            data,
            decision,
            0,
            100,
        )

        self.assertIsNone(trade)


if __name__ == "__main__":
    unittest.main(verbosity=2)
