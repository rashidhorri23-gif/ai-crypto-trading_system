import tempfile
import unittest
from pathlib import Path

from Src.Engines.AutonomousTechnicalAnalyst.binance_copy_trader_monitor import (
    compare_positions,
    run_monitor,
    select_top_traders,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def get_ranked_traders(self, period):
        self.calls.append(("ranking", period))
        if period == "30D":
            return [
                {"traderId": "A", "name": "Alpha", "sharpeRatio": 2.5,
                 "maxDrawdown": 0.10, "copierPnl": 100, "winRate": 0.7},
                {"traderId": "B", "name": "Beta", "sharpeRatio": 1.8,
                 "maxDrawdown": 0.20, "copierPnl": 50, "winRate": 0.6},
            ]
        return [
            {"traderId": "B", "name": "Beta", "sharpeRatio": 2.0,
             "maxDrawdown": 0.20, "copierPnl": 50, "winRate": 0.6},
            {"traderId": "A", "name": "Alpha", "sharpeRatio": 2.2,
             "maxDrawdown": 0.10, "copierPnl": 100, "winRate": 0.7},
        ]

    def get_open_positions(self, trader_id):
        self.calls.append(("positions", trader_id))
        return {"positions": [], "hidden": False}


class BinanceCopyTraderMonitorTests(unittest.TestCase):
    def test_selection_blends_periods_and_filters_risk(self):
        selected = select_top_traders(
            [
                {"traderId": "A", "sharpeRatio": 2, "maxDrawdown": 0.1, "copierPnl": 10},
                {"traderId": "B", "sharpeRatio": 1, "maxDrawdown": 0.7, "copierPnl": 10},
            ],
            [
                {"traderId": "A", "sharpeRatio": 2, "maxDrawdown": 0.1, "copierPnl": 10},
                {"traderId": "B", "sharpeRatio": 1, "maxDrawdown": 0.7, "copierPnl": 10},
            ],
        )
        self.assertEqual([item["trader_id"] for item in selected], ["A"])

    def test_hidden_snapshot_does_not_generate_false_closures(self):
        old = {"BTCUSDT:LONG": {"symbol": "BTCUSDT", "side": "LONG", "size": 1}}
        self.assertEqual(compare_positions(old, {}, hidden=True), [])

    def test_position_changes_are_detected(self):
        old = {"BTCUSDT:LONG": {"symbol": "BTCUSDT", "side": "LONG", "size": 1}}
        increased = {"BTCUSDT:LONG": {"symbol": "BTCUSDT", "side": "LONG", "size": 2}}
        events = compare_positions(old, increased)
        self.assertEqual([event[0] for event in events], ["INCREASED"])
        events = compare_positions(increased, {})
        self.assertEqual([event[0] for event in events], ["CLOSED"])

    def test_first_run_baselines_without_false_open_alerts(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state, messages = run_monitor(client=client, state_path=state_path, notify=False)
            self.assertTrue(state["initialized"])
            self.assertEqual(messages, [])
            self.assertEqual(len([call for call in client.calls if call[0] == "positions"]), 2)


if __name__ == "__main__":
    unittest.main()
