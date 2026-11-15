
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from . import ai_backtest


class TestBacktestIntegration(unittest.TestCase):

    def test_run_backtest_processes_ai_decision_and_saves_outputs(self):

        data = pd.DataFrame({
            "timestamp_utc": [
                f"2026-11-01 00:{i:02d}:00+00:00"
                for i in range(8)
            ]
        })

        long_decision = {
            "decision": "LONG",
            "bias": "BULLISH",
            "confidence": 85,
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "risk_reward": 2.0,
            "reason": "Integration test",
        }

        wait_decision = {
            "decision": "WAIT",
            "bias": "NEUTRAL",
            "confidence": 50,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": None,
            "reason": "No valid setup",
        }

        trade = {
            "direction": "LONG",
            "entry_price": 100.0,
            "exit_price": 110.0,
            "net_r": 1.8,
            "exit_reason": "TAKE_PROFIT",
            "exit_index": 3,
        }

        ai_results = [
            (
                {
                    "status": "success",
                    "model": "test-model",
                    "decision": long_decision,
                },
                False,
            ),
            (
                {
                    "status": "success",
                    "model": "test-model",
                    "decision": wait_decision,
                },
                False,
            ),
        ]

        original_directory = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_directory:
            try:
                os.chdir(temp_directory)

                with (
                    patch.object(ai_backtest.Config, "validate"),
                    patch.object(
                        ai_backtest,
                        "prepare_market_features",
                        return_value=data,
                    ) as mock_prepare,
                    patch.object(
                        ai_backtest,
                        "validate_backtest_data",
                    ),
                    patch.object(
                        ai_backtest,
                        "load_cache",
                        return_value={},
                    ),
                    patch.object(
                        ai_backtest,
                        "save_cache",
                    ),
                    patch.object(
                        ai_backtest,
                        "build_prompt",
                        return_value="test prompt",
                    ),
                    patch.object(
                        ai_backtest,
                        "get_ai_decision",
                        side_effect=ai_results,
                    ),
                    patch.object(
                        ai_backtest,
                        "find_entry_fill",
                        return_value=(3, 100.0),
                    ),
                    patch.object(
                        ai_backtest,
                        "execute_trade",
                        return_value=trade,
                    ) as mock_execute,
                    patch.object(
                        ai_backtest,
                        "WARMUP_CANDLES",
                        2,
                    ),
                    patch.object(
                        ai_backtest,
                        "SIGNAL_INTERVAL",
                        1,
                    ),
                    patch.object(
                        ai_backtest,
                        "ENTRY_VALIDITY_CANDLES",
                        1,
                    ),
                    patch.object(
                        ai_backtest,
                        "MAX_HOLDING_CANDLES",
                        2,
                    ),
                    patch.object(
                        ai_backtest,
                        "MAX_AI_CALLS",
                        10,
                    ),
                ):

                    summary = ai_backtest.run_backtest()

                    expected_directory = Path("Data/Backtest")

                    self.assertTrue(
                        (
                            expected_directory
                            / "ernie_backtest_trades.csv"
                        ).exists()
                    )

                    self.assertTrue(
                        (
                            expected_directory
                            / "ernie_backtest_decisions.csv"
                        ).exists()
                    )

                    self.assertTrue(
                        (
                            expected_directory
                            / "ernie_backtest_summary.json"
                        ).exists()
                    )

                    self.assertEqual(summary["api_calls"], 2)
                    self.assertEqual(summary["api_errors"], 0)
                    self.assertEqual(summary["processing_errors"], 0)

                    self.assertEqual(
                        summary["overall"]["total_trades"],
                        1,
                    )

                    self.assertEqual(
                        summary["long"]["total_trades"],
                        1,
                    )

                    self.assertEqual(
                        summary["long"]["net_r"],
                        1.8,
                    )

                    self.assertEqual(
                        summary["short"]["total_trades"],
                        0,
                    )

                    self.assertEqual(
                        summary["short"]["net_r"],
                        0.0,
                    )

                    self.assertEqual(
                        mock_execute.call_count,
                        1,
                    )

                    self.assertEqual(
                        mock_prepare.call_count,
                        1,
                    )

            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
