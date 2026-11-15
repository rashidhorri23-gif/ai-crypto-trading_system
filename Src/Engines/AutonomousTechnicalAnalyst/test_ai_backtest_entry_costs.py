
import unittest

import pandas as pd

from Src.Engines.AutonomousTechnicalAnalyst.ai_backtest import (
    find_entry_fill,
    execute_trade,
    ENTRY_VALIDITY_CANDLES,
    MAX_HOLDING_CANDLES,
    FEE_RATE_PER_SIDE,
    SLIPPAGE_RATE_PER_SIDE,
)


def make_market_data(candles):
    rows = []

    for index, candle in enumerate(candles):
        rows.append({
            "timestamp_utc": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=4 * index),
            "open": candle.get("open", 100.0),
            "high": candle["high"],
            "low": candle["low"],
            "close": candle.get("close", 100.0),
            "volume": 1000.0,
        })

    return pd.DataFrame(rows)


class TestEntryFill(unittest.TestCase):

    def setUp(self):
        self.decision = {
            "decision": "LONG",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
        }

    def test_entry_fills_on_first_valid_candle(self):
        data = make_market_data([
            {"high": 105, "low": 101},
            {"high": 102, "low": 99},
            {"high": 105, "low": 101},
        ])

        index, price = find_entry_fill(data, 0, self.decision)

        self.assertEqual(index, 1)
        self.assertEqual(price, 100.0)

    def test_entry_fills_on_last_valid_candle(self):
        data = make_market_data([
            {"high": 105, "low": 101},
            {"high": 105, "low": 101},
            {"high": 105, "low": 101},
            {"high": 102, "low": 99},
        ])

        index, price = find_entry_fill(data, 0, self.decision)

        self.assertEqual(index, 3)
        self.assertEqual(price, 100.0)

    def test_entry_does_not_fill_after_validity_expires(self):
        data = make_market_data([
            {"high": 105, "low": 101},
            {"high": 105, "low": 101},
            {"high": 105, "low": 101},
            {"high": 105, "low": 101},
            {"high": 102, "low": 99},
        ])

        index, price = find_entry_fill(data, 0, self.decision)

        self.assertIsNone(index)
        self.assertIsNone(price)

    def test_entry_uses_first_matching_candle(self):
        data = make_market_data([
            {"high": 105, "low": 101},
            {"high": 102, "low": 99},
            {"high": 102, "low": 99},
            {"high": 102, "low": 99},
        ])

        index, price = find_entry_fill(data, 0, self.decision)

        self.assertEqual(index, 1)
        self.assertEqual(price, 100.0)

    def test_entry_skips_invalid_candle_values(self):
        data = make_market_data([
            {"high": 105, "low": 101},
            {"high": float("nan"), "low": 99},
            {"high": 102, "low": 99},
        ])

        index, price = find_entry_fill(data, 0, self.decision)

        self.assertEqual(index, 2)
        self.assertEqual(price, 100.0)


class TestExecutionCosts(unittest.TestCase):

    def test_long_trade_cost_calculation(self):
        data = make_market_data([
            {"high": 101, "low": 99, "close": 100},
            {"high": 111, "low": 101, "close": 108},
        ])

        decision = {
            "decision": "LONG",
            "decision_index": 0,
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "confidence": 80,
            "reason": "Test",
        }

        trade = execute_trade(data, decision, 0, 100.0)

        self.assertIsNotNone(trade)

        cost = FEE_RATE_PER_SIDE + SLIPPAGE_RATE_PER_SIDE

        expected_entry = 100.0 * (1 + cost)
        expected_exit = 110.0 * (1 - cost)
        expected_net_pnl = expected_exit - expected_entry
        expected_initial_risk = expected_entry - 95.0
        expected_net_r = expected_net_pnl / expected_initial_risk

        self.assertAlmostEqual(
            trade["executed_entry"],
            expected_entry,
            places=8,
        )

        self.assertAlmostEqual(
            trade["executed_exit"],
            expected_exit,
            places=8,
        )

        self.assertAlmostEqual(
            trade["net_pnl_per_unit"],
            expected_net_pnl,
            places=8,
        )

        self.assertAlmostEqual(
            trade["initial_risk"],
            expected_initial_risk,
            places=8,
        )

        self.assertAlmostEqual(
            trade["net_r"],
            expected_net_r,
            places=8,
        )

    def test_short_trade_cost_calculation(self):
        data = make_market_data([
            {"high": 101, "low": 99, "close": 100},
            {"high": 99, "low": 89, "close": 92},
        ])

        decision = {
            "decision": "SHORT",
            "decision_index": 0,
            "entry": 100.0,
            "stop_loss": 105.0,
            "take_profit": 90.0,
            "confidence": 80,
            "reason": "Test",
        }

        trade = execute_trade(data, decision, 0, 100.0)

        self.assertIsNotNone(trade)

        cost = FEE_RATE_PER_SIDE + SLIPPAGE_RATE_PER_SIDE

        expected_entry = 100.0 * (1 - cost)
        expected_exit = 90.0 * (1 + cost)
        expected_net_pnl = expected_entry - expected_exit
        expected_initial_risk = 105.0 - expected_entry
        expected_net_r = expected_net_pnl / expected_initial_risk

        self.assertAlmostEqual(
            trade["executed_entry"],
            expected_entry,
            places=8,
        )

        self.assertAlmostEqual(
            trade["executed_exit"],
            expected_exit,
            places=8,
        )

        self.assertAlmostEqual(
            trade["net_pnl_per_unit"],
            expected_net_pnl,
            places=8,
        )

        self.assertAlmostEqual(
            trade["net_r"],
            expected_net_r,
            places=8,
        )


class TestMaximumHoldingPeriod(unittest.TestCase):

    def test_trade_closes_after_maximum_holding_period(self):
        candles = []

        for index in range(MAX_HOLDING_CANDLES + 2):
            candles.append({
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
            })

        data = make_market_data(candles)

        decision = {
            "decision": "LONG",
            "decision_index": 0,
            "entry": 100.0,
            "stop_loss": 90.0,
            "take_profit": 120.0,
            "confidence": 80,
            "reason": "Test",
        }

        trade = execute_trade(data, decision, 0, 100.0)

        self.assertIsNotNone(trade)
        self.assertEqual(
            trade["exit_index"],
            MAX_HOLDING_CANDLES,
        )
        self.assertEqual(
            trade["exit_reason"],
            "TIME_EXIT",
        )

    def test_holding_period_constant_is_24(self):
        self.assertEqual(MAX_HOLDING_CANDLES, 24)

    def test_entry_validity_constant_is_three(self):
        self.assertEqual(ENTRY_VALIDITY_CANDLES, 3)


if __name__ == "__main__":
    unittest.main()
