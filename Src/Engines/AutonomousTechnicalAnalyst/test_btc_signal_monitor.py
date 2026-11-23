
"""Unit tests for the AI-report-driven BTC paper-trading monitor.

Tests use synthetic in-memory ledgers and mocked persistence.
They never access the repository's live paper-trading ledger.
"""

import copy
import time
import unittest
from unittest.mock import patch

from Src.Engines.AutonomousTechnicalAnalyst import monitor_btc_signal as monitor
from Src.Engines.AutonomousTechnicalAnalyst import paper_trading


def make_report(
    decision="WAIT",
    analysis_id="unit-test-001",
    open_trade_action="NONE",
    previous_signal_action="NONE",
):
    report = {
        "schema_version": "2.0",
        "analysis_id": analysis_id,
        "generated_at": monitor.utc_now().isoformat(),
        "symbol": "BTC-USDT",
        "timeframes": ["4H", "15M", "5M"],
        "decision": decision,
        "confidence": 80,
        "market_bias": "NEUTRAL",
        "entry": {
            "entry_type": "NONE",
            "entry_price": None,
            "entry_zone_low": None,
            "entry_zone_high": None,
            "entry_expires_at": None,
        },
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "take_profit_3": None,
        "reasoning": "Unit test.",
        "main_risks": [],
        "data_limitations": [],
        "previous_signal_action": previous_signal_action,
        "open_trade_action": open_trade_action,
        "open_trade_updated_levels": {
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "take_profit_3": None,
        },
    }

    if decision == "LONG":
        report.update({
            "market_bias": "BULLISH",
            "entry": {
                "entry_type": "MARKET",
                "entry_price": 100.0,
                "entry_zone_low": 99.0,
                "entry_zone_high": 101.0,
                "entry_expires_at": None,
            },
            "stop_loss": 95.0,
            "take_profit_1": 105.0,
            "take_profit_2": 110.0,
            "take_profit_3": 115.0,
        })

    elif decision == "SHORT":
        report.update({
            "market_bias": "BEARISH",
            "entry": {
                "entry_type": "LIMIT",
                "entry_price": 100.0,
                "entry_zone_low": 99.0,
                "entry_zone_high": 101.0,
                "entry_expires_at": monitor.utc_iso(time.time() + 3600),
            },
            "stop_loss": 105.0,
            "take_profit_1": 95.0,
            "take_profit_2": 90.0,
            "take_profit_3": 85.0,
        })

    return report


def make_open_trade(direction="SHORT", trailing_active=False):
    ledger = {
        "version": 1,
        "settings": paper_trading.settings(),
        "trades": [],
    }

    if direction == "SHORT":
        levels = (105.0, 95.0, 90.0, 85.0)
    else:
        levels = (95.0, 105.0, 110.0, 115.0)

    trade, created = paper_trading.open_trade_from_ai(
        ledger=ledger,
        direction=direction,
        entry_type="MARKET",
        entry_price=100.0,
        stop_loss=levels[0],
        take_profit_1=levels[1],
        take_profit_2=levels[2],
        take_profit_3=levels[3],
        signal_fingerprint=f"synthetic-{direction}",
        timestamp=1000,
    )
    assert created

    if trailing_active:
        trade["trailing_active"] = True
        trade["stop_loss_current"] = (
            101.0 if direction == "SHORT" else 99.0
        )

    return ledger, trade


class TestReportValidation(unittest.TestCase):

    def test_valid_wait(self):
        self.assertIsNone(monitor.validate_ai_plan(make_report()))

    def test_valid_long(self):
        plan = monitor.validate_ai_plan(make_report("LONG"))
        self.assertEqual(plan["direction"], "LONG")
        self.assertEqual(plan["entry"]["price"], 100.0)

    def test_valid_short_limit_with_expiry(self):
        plan = monitor.validate_ai_plan(make_report("SHORT"))
        self.assertEqual(plan["direction"], "SHORT")
        self.assertEqual(plan["entry"]["type"], "LIMIT")

    def test_wait_rejects_non_null_trade_level(self):
        report = make_report()
        report["take_profit_1"] = 101.0
        with self.assertRaises(ValueError):
            monitor.validate_ai_plan(report)

    def test_long_rejects_invalid_stop(self):
        report = make_report("LONG")
        report["stop_loss"] = 101.0
        with self.assertRaises(ValueError):
            monitor.validate_ai_plan(report)

    def test_entry_must_be_inside_zone(self):
        report = make_report("LONG")
        report["entry"]["entry_price"] = 102.0
        with self.assertRaises(ValueError):
            monitor.validate_ai_plan(report)

    def test_limit_rejects_expired_entry(self):
        report = make_report("SHORT")
        report["entry"]["entry_expires_at"] = monitor.utc_iso(time.time() - 60)
        with self.assertRaises(ValueError):
            monitor.validate_ai_plan(report)

    def test_unknown_decision_is_rejected(self):
        report = make_report()
        report["decision"] = "BUY"
        with self.assertRaises(ValueError):
            monitor.validate_ai_plan(report)


class TestAiReportProcessing(unittest.TestCase):

    def setUp(self):
        self.state = {
            "version": 2,
            "last_analysis_id": None,
            "last_trade_action_analysis_id": None,
            "pending_signal": None,
            "pending_ai_close": None,
            "notification_ids": [],
        }
        self.ledger = {
            "version": 1,
            "settings": paper_trading.settings(),
            "trades": [],
        }

    def run_report(self, report):
        generated = monitor.parse_timestamp(
            report["generated_at"], "generated_at"
        )
        with patch.object(monitor, "save_state"), \
             patch.object(monitor, "save_current_signal"), \
             patch.object(monitor, "send_telegram", return_value=True), \
             patch.object(paper_trading, "save_ledger"):
            monitor.process_new_ai_report(
                report,
                report["analysis_id"],
                generated,
                self.state,
                self.ledger,
            )

    def test_new_long_creates_pending_signal(self):
        report = make_report("LONG")
        self.run_report(report)
        self.assertEqual(self.state["pending_signal"]["direction"], "LONG")
        self.assertEqual(self.state["pending_signal"]["status"], "PENDING")

    def test_duplicate_analysis_id_is_ignored(self):
        report = make_report("LONG")
        self.state["last_analysis_id"] = report["analysis_id"]
        before = copy.deepcopy(self.state)
        with patch.object(monitor, "save_state") as save_mock:
            monitor.process_new_ai_report(
                report,
                report["analysis_id"],
                monitor.parse_timestamp(report["generated_at"], "generated_at"),
                self.state,
                self.ledger,
            )
        self.assertEqual(self.state, before)
        save_mock.assert_not_called()

    def test_invalid_report_does_not_change_state_or_ledger(self):
        report = make_report("LONG")
        report["stop_loss"] = 101.0
        state_before = copy.deepcopy(self.state)
        ledger_before = copy.deepcopy(self.ledger)
        with self.assertRaises(ValueError):
            monitor.process_new_ai_report(
                report,
                report["analysis_id"],
                monitor.parse_timestamp(report["generated_at"], "generated_at"),
                self.state,
                self.ledger,
            )
        self.assertEqual(self.state, state_before)
        self.assertEqual(self.ledger, ledger_before)

    def test_ai_close_is_queued_not_executed_immediately(self):
        ledger, trade = make_open_trade("SHORT")
        self.ledger = ledger
        report = make_report(
            "WAIT",
            analysis_id="close-request-001",
            open_trade_action="CLOSE",
        )
        self.run_report(report)
        self.assertEqual(trade["status"], "OPEN")
        self.assertEqual(
            self.state["pending_ai_close"]["analysis_id"],
            "close-request-001",
        )

    def test_valid_short_trailing_stop_update(self):
        ledger, trade = make_open_trade("SHORT", trailing_active=True)
        self.ledger = ledger
        report = make_report(
            "WAIT",
            analysis_id="update-levels-001",
            open_trade_action="UPDATE_LEVELS",
        )
        report["open_trade_updated_levels"] = {
            "stop_loss": 100.5,
            "take_profit_1": 94.0,
            "take_profit_2": None,
            "take_profit_3": None,
        }
        self.run_report(report)
        self.assertEqual(trade["stop_loss_current"], 100.5)
        self.assertEqual(trade["levels_updated_by"], "AI")

    def test_invalid_short_trailing_stop_update_is_atomic(self):
        ledger, trade = make_open_trade("SHORT", trailing_active=True)
        self.ledger = ledger
        trade_before = copy.deepcopy(trade)
        state_before = copy.deepcopy(self.state)
        report = make_report(
            "WAIT",
            analysis_id="bad-update-001",
            open_trade_action="UPDATE_LEVELS",
        )
        report["open_trade_updated_levels"] = {
            "stop_loss": 104.0,
            "take_profit_1": None,
            "take_profit_2": None,
            "take_profit_3": None,
        }
        with self.assertRaises(ValueError):
            monitor.process_new_ai_report(
                report,
                report["analysis_id"],
                monitor.parse_timestamp(report["generated_at"], "generated_at"),
                self.state,
                self.ledger,
            )
        self.assertEqual(trade, trade_before)
        self.assertEqual(self.state, state_before)


class TestCandleExecution(unittest.TestCase):

    def test_ai_close_waits_for_candle_after_report_time(self):
        ledger, trade = make_open_trade("SHORT")
        report_time = 2000
        state = {
            "pending_ai_close": {
                "analysis_id": "close-001",
                "generated_at_timestamp": report_time,
                "trade_ids": [trade["trade_id"]],
            }
        }
        candle = {
            "timestamp": report_time - 300,
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 10.0,
        }
        with patch.object(paper_trading, "save_ledger"), \
             patch.object(monitor, "save_state"), \
             patch.object(monitor, "notify_closed_trades"):
            monitor.process_ai_close(candle, state, ledger)
        self.assertEqual(trade["status"], "OPEN")
        self.assertIsNotNone(state["pending_ai_close"])

    def test_ai_close_executes_on_later_candle(self):
        ledger, trade = make_open_trade("SHORT")
        state = {
            "pending_ai_close": {
                "analysis_id": "close-002",
                "generated_at_timestamp": 2000,
                "trade_ids": [trade["trade_id"]],
            }
        }
        candle = {
            "timestamp": 2000,
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 99.5, "volume": 10.0,
        }
        with patch.object(paper_trading, "save_ledger"), \
             patch.object(monitor, "save_state"), \
             patch.object(monitor, "notify_closed_trades"):
            monitor.process_ai_close(candle, state, ledger)
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["exit_reason"], "AI_CLOSE")
        self.assertIsNone(state["pending_ai_close"])
        self.assertEqual(state["last_completed_ai_close_id"], "close-002")

    def test_stop_loss_has_priority_if_sl_and_tp1_hit_together(self):
        ledger, trade = make_open_trade("LONG")
        candle = {
            "timestamp": 1100,
            "open": 100.0, "high": 106.0, "low": 94.0,
            "close": 100.0, "volume": 10.0,
        }
        paper_trading.process_candle(ledger, candle)
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["exit_reason"], "STOP_LOSS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
