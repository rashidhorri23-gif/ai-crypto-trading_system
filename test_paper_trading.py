import tempfile
import unittest
from pathlib import Path

from Src.Engines.AutonomousTechnicalAnalyst import paper_trading as paper


class TestPaperTrading(unittest.TestCase):
    def setUp(self):
        self.ledger = {"version": 1, "settings": paper.settings(), "trades": []}
        self.long_plan = {
            "stop_loss": 95.0,
            "take_profit_1": 110.0,
            "take_profit_2": 115.0,
            "take_profit_3": 120.0,
        }
        self.short_plan = {
            "stop_loss": 105.0,
            "take_profit_1": 90.0,
            "take_profit_2": 85.0,
            "take_profit_3": 80.0,
        }

    def candle(self, timestamp, open_price, high, low, close):
        return {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }

    def test_open_trade_uses_1000_margin_and_3x_notional(self):
        trade, created = paper.open_trade(
            self.ledger, "LONG", self.long_plan,
            self.candle(1000, 104, 106, 103, 105), "fingerprint-1",
        )
        self.assertTrue(created)
        self.assertEqual(trade["margin_usd"], 1000.0)
        self.assertEqual(trade["notional_usd"], 3000.0)
        self.assertEqual(trade["status"], "OPEN")

    def test_duplicate_signal_does_not_create_second_trade(self):
        candle = self.candle(1000, 104, 106, 103, 105)
        first, created_first = paper.open_trade(
            self.ledger, "LONG", self.long_plan, candle, "same-fingerprint",
        )
        second, created_second = paper.open_trade(
            self.ledger, "LONG", self.long_plan, candle, "same-fingerprint",
        )
        self.assertTrue(created_first)
        self.assertFalse(created_second)
        self.assertIs(first, second)
        self.assertEqual(len(self.ledger["trades"]), 1)

    def test_tp1_closes_full_trade(self):
        trade, _ = paper.open_trade(
            self.ledger, "LONG", self.long_plan,
            self.candle(1000, 104, 106, 103, 105), "tp1-test",
        )
        paper.process_candle(
            self.ledger, self.candle(1600, 108, 111, 107, 110),
        )
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT_1_FULL_EXIT")
        self.assertGreater(trade["realized_pnl_usd"], 0)

    def test_stop_wins_when_stop_and_tp1_touch_same_candle(self):
        trade, _ = paper.open_trade(
            self.ledger, "LONG", self.long_plan,
            self.candle(1000, 104, 106, 103, 105), "same-candle-test",
        )
        paper.process_candle(
            self.ledger, self.candle(1600, 105, 111, 94, 100),
        )
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")
        self.assertEqual(trade["exit_price"], 95.0)

    def test_trailing_stop_activates_at_70_net_and_locks_50(self):
        trade, _ = paper.open_trade(
            self.ledger, "LONG", self.long_plan,
            self.candle(1000, 104, 106, 103, 105), "trailing-test",
        )
        paper.process_candle(
            self.ledger, self.candle(1600, 105, 108, 104, 107),
        )
        self.assertTrue(trade["trailing_active"])
        locked = paper._net_at_exit(trade, trade["stop_loss_current"])
        self.assertAlmostEqual(locked, 50.0, places=5)

    def test_trailing_giveback_is_unset_pending_user_decision(self):
        self.assertIsNone(paper.TRAIL_GIVEBACK_NET_USD)
        self.assertIsNone(paper.settings()["trailing_giveback_net_usd"])

    def test_short_stop_and_tp1(self):
        trade, _ = paper.open_trade(
            self.ledger, "SHORT", self.short_plan,
            self.candle(1000, 99, 100, 97, 98), "short-tp1-test",
        )
        paper.process_candle(
            self.ledger, self.candle(1600, 94, 95, 89, 90),
        )
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["exit_reason"], "TAKE_PROFIT_1_FULL_EXIT")

    def test_ledger_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper_trading.json"
            paper.save_ledger(self.ledger, path)
            loaded = paper.load_ledger(path)
            self.assertEqual(loaded["trades"], [])
            self.assertEqual(loaded["settings"]["notional_usd_per_trade"], 3000.0)


if __name__ == "__main__":
    unittest.main()
