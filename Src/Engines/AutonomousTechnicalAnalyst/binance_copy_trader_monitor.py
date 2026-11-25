"""Monitor public Binance Futures Copy Trading portfolios through Kopyon.

This module observes public position snapshots only. It never places orders.
The free-plan mode polls once per hour to keep estimated usage below 10,000
requests per 30-day month (2 leaderboard calls + up to 10 position calls/hour).
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import requests

API_BASE = "https://api.kopyon.com/v5/binance-copy-trading"
STATE_PATH = Path("Data/Signals/binance_copy_trader_state.json")
TOP_N = 10
RANKING_ROWS = 50
REQUEST_TIMEOUT = 30


class KopyonClient:
    def __init__(self, api_key, session=None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def get(self, endpoint, params=None):
        response = self.session.get(
            f"{API_BASE}/{endpoint}",
            headers={"X-API-KEY": self.api_key},
            params=params or {},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Kopyon API error: {payload['error']}")
        return payload.get("data") or {}

    def get_ranked_traders(self, period):
        data = self.get(
            "traders",
            {"period": period, "sortBy": "sharpeRatio", "order": "DESC",
             "page": 1, "rows": RANKING_ROWS},
        )
        return data.get("traders") or []

    def get_open_positions(self, trader_id):
        return self.get("openPositions", {"traderId": trader_id})


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_top_traders(rankings_30d, rankings_90d, limit=TOP_N):
    """Blend 30D and 90D leaderboard ranks, then apply risk sanity checks."""
    candidates = {}

    for period_weight, traders in ((0.60, rankings_30d), (0.40, rankings_90d)):
        for rank, trader in enumerate(traders, start=1):
            trader_id = str(trader.get("traderId") or "").strip()
            if not trader_id:
                continue
            item = candidates.setdefault(
                trader_id,
                {"trader": trader, "weighted_rank": 0.0, "weight_seen": 0.0},
            )
            item["weighted_rank"] += rank * period_weight
            item["weight_seen"] += period_weight
            if _number(trader.get("maxDrawdown"), 0.0) > _number(
                item["trader"].get("maxDrawdown"), 0.0
            ):
                item["trader"] = trader

    eligible = []
    for trader_id, item in candidates.items():
        trader = item["trader"]
        sharpe = _number(trader.get("sharpeRatio"), 0.0)
        drawdown = _number(trader.get("maxDrawdown"), 0.0)
        copier_pnl = trader.get("copierPnl")
        if sharpe <= 0 or drawdown > 0.50:
            continue
        if copier_pnl is not None and _number(copier_pnl) <= 0:
            continue
        average_rank = item["weighted_rank"] / max(item["weight_seen"], 0.01)
        eligible.append((average_rank, trader_id, trader))

    eligible.sort(key=lambda row: (row[0], -_number(row[2].get("sharpeRatio"))))
    selected = []
    for rank, (_, trader_id, trader) in enumerate(eligible[:limit], start=1):
        selected.append({
            "rank": rank,
            "trader_id": trader_id,
            "name": trader.get("name") or f"Trader {trader_id[-6:]}",
            "sharpe_ratio": trader.get("sharpeRatio"),
            "max_drawdown": trader.get("maxDrawdown"),
            "win_rate": trader.get("winRate"),
            "copier_pnl": trader.get("copierPnl"),
            "roi": trader.get("roi"),
        })
    return selected


def _position_key(position):
    position_id = position.get("positionId")
    if position_id:
        return str(position_id)
    return f"{position.get('symbol', '')}:{position.get('side', '')}"


def compare_positions(previous, current, hidden=False):
    """Return position events; never infer closure from a hidden snapshot."""
    if hidden:
        return []
    events = []
    previous = previous or {}
    current = current or {}

    for key, position in current.items():
        old = previous.get(key)
        if old is None:
            events.append(("OPENED", position))
            continue
        old_size = abs(_number(old.get("size")))
        new_size = abs(_number(position.get("size")))
        if new_size > old_size + 1e-12:
            events.append(("INCREASED", position))
        elif new_size < old_size - 1e-12:
            events.append(("REDUCED", position))

    for key, position in previous.items():
        if key not in current:
            events.append(("CLOSED", position))
    return events


def _index_positions(positions):
    return {_position_key(position): position for position in positions or []}


def load_state(path=STATE_PATH):
    if not path.exists():
        return {"version": 1, "initialized": False, "traders": {}, "updated_at": None}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state, path=STATE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _format_position_event(event_type, trader, position):
    labels = {
        "OPENED": "NEW POSITION",
        "INCREASED": "POSITION INCREASED",
        "REDUCED": "POSITION REDUCED",
        "CLOSED": "POSITION CLOSED",
    }
    side = position.get("side", "UNKNOWN")
    symbol = position.get("symbol", "UNKNOWN")
    lines = [
        f"Binance Futures | {labels[event_type]}",
        f"Trader: {trader.get('name')} (Top #{trader.get('rank')})",
        f"Market: {symbol} | Side: {side}",
    ]
    for field, label in (
        ("entryPrice", "Entry"),
        ("markPrice", "Mark"),
        ("size", "Size"),
        ("leverage", "Leverage"),
        ("unrealizedPnl", "Unrealized PnL"),
    ):
        value = position.get(field)
        if value is not None:
            lines.append(f"{label}: {value}")
    lines.append("Observed from public position data; execution time may be delayed.")
    return "\n".join(lines)


def send_telegram_message(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID secret.")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": text},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram rejected the message: {payload}")
    return payload


def run_monitor(client=None, state_path=STATE_PATH, notify=True):
    api_key = os.getenv("KOPYON_API_KEY", "").strip()
    if client is None:
        if not api_key:
            raise RuntimeError("Missing KOPYON_API_KEY GitHub Actions secret.")
        client = KopyonClient(api_key)

    ranking_30d = client.get_ranked_traders("30D")
    ranking_90d = client.get_ranked_traders("90D")
    selected = select_top_traders(ranking_30d, ranking_90d)
    if len(selected) < TOP_N:
        print(f"WARNING: only {len(selected)} eligible traders found; target is {TOP_N}.")

    state = load_state(state_path)
    previous_traders = state.get("traders") or {}
    next_traders = {}
    messages = []

    for trader in selected:
        trader_id = trader["trader_id"]
        snapshot = client.get_open_positions(trader_id)
        positions = snapshot.get("positions") or []
        hidden = bool(snapshot.get("hidden", False))
        current_index = _index_positions(positions)
        previous_entry = previous_traders.get(trader_id) or {}
        previous_index = previous_entry.get("positions") or {}

        if state.get("initialized"):
            for event_type, position in compare_positions(
                previous_index, current_index, hidden=hidden
            ):
                messages.append(_format_position_event(event_type, trader, position))

        next_traders[trader_id] = {
            "trader": trader,
            "positions": previous_index if hidden else current_index,
            "hidden": hidden,
        }

    now = datetime.now(timezone.utc).isoformat()
    next_state = {
        "version": 1,
        "initialized": True,
        "updated_at": now,
        "selection": selected,
        "traders": next_traders,
    }
    save_state(next_state, state_path)

    if notify:
        for message in messages:
            send_telegram_message(message)

    print(json.dumps({
        "status": "ok",
        "selected_traders": len(selected),
        "position_alerts": len(messages),
        "hidden_portfolios": sum(1 for item in next_traders.values() if item["hidden"]),
        "updated_at": now,
    }, ensure_ascii=False, indent=2))
    return next_state, messages


if __name__ == "__main__":
    run_monitor()
