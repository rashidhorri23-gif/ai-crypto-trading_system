
import math
import unittest

from .ai_backtest import validate_decision, calculate_performance


class TestDecisionValidation(unittest.TestCase):

    def test_valid_long_decision(self):
        payload = {
            "decision": "LONG",
            "bias": "BULLISH",
            "confidence": 85,
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "risk_reward": 999,
            "reason": "Test long",
            "invalidation": "Price closes below stop",
        }

        result = validate_decision(payload)

        self.assertEqual(result["decision"], "LONG")
        self.assertAlmostEqual(result["risk_reward"], 2.0)

    def test_valid_short_decision(self):
        payload = {
            "decision": "SHORT",
            "bias": "BEARISH",
            "confidence": 80,
            "entry": 100,
            "stop_loss": 105,
            "take_profit": 90,
            "risk_reward": 999,
            "reason": "Test short",
            "invalidation": "Price closes above stop",
        }

        result = validate_decision(payload)

        self.assertEqual(result["decision"], "SHORT")
        self.assertAlmostEqual(result["risk_reward"], 2.0)

    def test_wait_decision(self):
        payload = {
            "decision": "WAIT",
            "bias": "NEUTRAL",
            "confidence": 50,
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 110,
            "risk_reward": 2,
            "reason": "No clear setup",
            "invalidation": "",
        }

        result = validate_decision(payload)

        self.assertEqual(result["decision"], "WAIT")
        self.assertEqual(result["entry"], 0)
        self.assertEqual(result["stop_loss"], 0)
        self.assertEqual(result["take_profit"], 0)
        self.assertEqual(result["risk_reward"], 0)

    def test_reject_non_dictionary_payload(self):
        with self.assertRaises(ValueError):
            validate_decision("LONG")

    def test_reject_invalid_decision(self):
        payload = {
            "decision": "BUY",
            "bias": "BULLISH",
            "confidence": 80,
        }

        with self.assertRaises(ValueError):
            validate_decision(payload)

    def test_reject_invalid_bias(self):
        payload = {
            "decision": "LONG",
            "bias": "UNKNOWN",
            "confidence": 80,
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 110,
        }

        with self.assertRaises(ValueError):
            validate_decision(payload)

    def test_reject_invalid_confidence(self):
        payload = {
            "decision": "LONG",
            "bias": "BULLISH",
            "confidence": 120,
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 110,
        }

        with self.assertRaises(ValueError):
            validate_decision(payload)

    def test_reject_invalid_long_geometry(self):
        payload = {
            "decision": "LONG",
            "bias": "BULLISH",
            "confidence": 80,
            "entry": 100,
            "stop_loss": 105,
            "take_profit": 110,
        }

        with self.assertRaises(ValueError):
            validate_decision(payload)

    def test_reject_invalid_short_geometry(self):
        payload = {
            "decision": "SHORT",
            "bias": "BEARISH",
            "confidence": 80,
            "entry": 100,
            "stop_loss": 95,
            "take_profit": 90,
        }

        with self.assertRaises(ValueError):
            validate_decision(payload)


class TestPerformanceCalculation(unittest.TestCase):

    def test_calculate_basic_performance(self):
        trades = [
            {"net_r": 1.0},
            {"net_r": -0.5},
            {"net_r": 0.0},
        ]

        result = calculate_performance(trades)

        self.assertEqual(result["total_trades"], 3)
        self.assertEqual(result["winning_trades"], 1)
        self.assertEqual(result["losing_trades"], 1)

        self.assertAlmostEqual(
            result["win_rate_percent"],
            100 / 3,
        )

        self.assertAlmostEqual(result["net_r"], 0.5)
        self.assertAlmostEqual(result["profit_factor"], 2.0)
        self.assertAlmostEqual(result["average_r"], 0.5 / 3)

        self.assertAlmostEqual(result["initial_equity"], 10000)
        self.assertAlmostEqual(result["final_equity"], 10049.5)

    def test_empty_trade_list(self):
        result = calculate_performance([])

        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["winning_trades"], 0)
        self.assertEqual(result["losing_trades"], 0)
        self.assertEqual(result["net_r"], 0)
        self.assertEqual(result["win_rate_percent"], 0)
        self.assertEqual(result["profit_factor"], 0)
        self.assertEqual(result["average_r"], 0)
        self.assertEqual(result["max_drawdown_percent"], 0)
        self.assertEqual(result["initial_equity"], 10000)
        self.assertEqual(result["final_equity"], 10000)

    def test_profit_factor_without_losses(self):
        trades = [
            {"net_r": 1.0},
            {"net_r": 2.0},
        ]

        result = calculate_performance(trades)

        self.assertEqual(result["winning_trades"], 2)
        self.assertEqual(result["losing_trades"], 0)
        self.assertTrue(math.isinf(result["profit_factor"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
