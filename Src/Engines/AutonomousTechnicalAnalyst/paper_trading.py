"""BTC-only paper trading ledger. No exchange orders are ever submitted."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRADING_FILE = ROOT / "Data" / "Trading" / "paper_trading.json"

MARGIN_USD = 1000.0
LEVERAGE = 3.0
NOTIONAL_USD = MARGIN_USD * LEVERAGE
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
TRAIL_TRIGGER_NET_USD = 70.0
TRAIL_LOCK_NET_USD = 50.0
TRAIL_GIVEBACK_NET_USD = None  # Deliberately unset pending user decision.
FIVE_MINUTES_SECONDS = 300


def utc_iso(timestamp=None):
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def load_ledger(path=None):
    path = path or TRADING_FILE
    if not path.exists():
        return {"version": 1, "settings": settings(), "trades": []}
    with path.open("r", encoding="utf-8") as handle:
        ledger = json.load(handle)
    if not isinstance(ledger, dict) or not isinstance(ledger.get("trades"), list):
        raise ValueError("Paper trading ledger has an invalid structure.")
    ledger.setdefault("version", 1)
    ledger.setdefault("settings", settings())
    return ledger


def save_ledger(ledger, path=None):
    path = path or TRADING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
    temporary.replace(path)


def settings():
    return {
        "margin_usd_per_trade": MARGIN_USD,
        "leverage": LEVERAGE,
        "notional_usd_per_trade": NOTIONAL_USD,
        "fee_rate_per_side": FEE_RATE,
        "slippage_rate_per_side": SLIPPAGE_RATE,
        "trailing_trigger_net_usd": TRAIL_TRIGGER_NET_USD,
        "trailing_lock_net_usd": TRAIL_LOCK_NET_USD,
        "trailing_giveback_net_usd": TRAIL_GIVEBACK_NET_USD,
        "tp1_exit_fraction": 1.0,
    }


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number.")
    return value


def _fills(direction, entry_price, exit_price):
    if direction == "LONG":
        return entry_price * (1 + SLIPPAGE_RATE), exit_price * (1 - SLIPPAGE_RATE)
    if direction == "SHORT":
        return entry_price * (1 - SLIPPAGE_RATE), exit_price * (1 + SLIPPAGE_RATE)
    raise ValueError("Direction must be LONG or SHORT.")


def trade_pnl(direction, quantity, entry_fill, exit_price):
    exit_fill = (
        exit_price * (1 - SLIPPAGE_RATE)
        if direction == "LONG"
        else exit_price * (1 + SLIPPAGE_RATE)
    )
    entry_notional = quantity * entry_fill
    exit_notional = quantity * exit_fill
    entry_fee = entry_notional * FEE_RATE
    exit_fee = exit_notional * FEE_RATE
    gross = quantity * (exit_fill - entry_fill)
    if direction == "SHORT":
        gross = -gross
    return {
        "gross_pnl_usd": gross,
        "entry_fee_usd": entry_fee,
        "exit_fee_usd": exit_fee,
        "total_fees_usd": entry_fee + exit_fee,
        "net_pnl_usd": gross - entry_fee - exit_fee,
        "exit_fill_price": exit_fill,
    }


def _net_at_exit(trade, exit_price):
    return trade_pnl(
        trade["direction"], trade["quantity"],
        trade["entry_fill_price"], exit_price,
    )["net_pnl_usd"]


def price_for_net_profit(trade, target_net):
    """Solve the exit trigger price after both-side fees and slippage."""
    entry = trade["entry_fill_price"]
    if trade["direction"] == "LONG":
        low, high = entry, entry * 2.0
        for _ in range(100):
            mid = (low + high) / 2
            if _net_at_exit(trade, mid) < target_net:
                low = mid
            else:
                high = mid
        return high
    low, high = entry * 0.01, entry
    for _ in range(100):
        mid = (low + high) / 2
        if _net_at_exit(trade, mid) >= target_net:
            high = mid
        else:
            low = mid
    return high


def open_trade(ledger, direction, plan, confirmation_candle, signal_fingerprint):
    """Create one idempotent paper trade from a confirmed 5M retest."""
    if direction not in ("LONG", "SHORT"):
        raise ValueError("Direction must be LONG or SHORT.")
    if not isinstance(confirmation_candle, dict):
        raise ValueError("Confirmation candle is required.")

    for existing in ledger["trades"]:
        if existing.get("signal_fingerprint") == signal_fingerprint:
            return existing, False

    entry_reference = _number(confirmation_candle.get("close"), "entry price")
    stop = _number(plan.get("stop_loss"), "stop_loss")
    tp1 = _number(plan.get("take_profit_1"), "take_profit_1")
    tp2 = _number(plan.get("take_profit_2"), "take_profit_2")
    tp3 = _number(plan.get("take_profit_3"), "take_profit_3")

    entry_fill, _ = _fills(direction, entry_reference, entry_reference)
    quantity = NOTIONAL_USD / entry_fill
    entry_fee = quantity * entry_fill * FEE_RATE
    timestamp = confirmation_candle["timestamp"] + FIVE_MINUTES_SECONDS

    trade = {
        "trade_id": signal_fingerprint,
        "signal_fingerprint": signal_fingerprint,
        "symbol": "BTC-USDT",
        "direction": direction,
        "status": "OPEN",
        "margin_usd": MARGIN_USD,
        "leverage": LEVERAGE,
        "notional_usd": NOTIONAL_USD,
        "quantity_btc": quantity,
        "quantity": quantity,
        "entry_reference_price": entry_reference,
        "entry_fill_price": entry_fill,
        "entry_timestamp": timestamp,
        "entry_time_utc": utc_iso(timestamp),
        "stop_loss_initial": stop,
        "stop_loss_current": stop,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "take_profit_3": tp3,
        "tp1_exit_fraction": 1.0,
        "tp2_reached": False,
        "tp3_reached": False,
        "trailing_active": False,
        "trailing_trigger_net_usd": TRAIL_TRIGGER_NET_USD,
        "trailing_lock_net_usd": TRAIL_LOCK_NET_USD,
        "trailing_giveback_net_usd": TRAIL_GIVEBACK_NET_USD,
        "entry_fee_usd": entry_fee,
        "exit_fee_usd": 0.0,
        "total_fees_usd": entry_fee,
        "gross_pnl_usd": 0.0,
        "realized_pnl_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "net_pnl_usd": 0.0,
        "exit_price": None,
        "exit_fill_price": None,
        "exit_timestamp": None,
        "exit_time_utc": None,
        "exit_reason": None,
        "created_at": utc_iso(),
    }
    ledger["trades"].append(trade)
    return trade, True


def _close_trade(trade, exit_price, timestamp, reason):
    result = trade_pnl(
        trade["direction"], trade["quantity"],
        trade["entry_fill_price"], exit_price,
    )
    trade.update({
        "status": "CLOSED",
        "exit_price": exit_price,
        "exit_fill_price": result["exit_fill_price"],
        "exit_timestamp": timestamp,
        "exit_time_utc": utc_iso(timestamp),
        "exit_reason": reason,
        "gross_pnl_usd": result["gross_pnl_usd"],
        "exit_fee_usd": result["exit_fee_usd"],
        "total_fees_usd": result["total_fees_usd"],
        "realized_pnl_usd": result["net_pnl_usd"],
        "unrealized_pnl_usd": 0.0,
        "net_pnl_usd": result["net_pnl_usd"],
        "last_updated_at": utc_iso(),
    })


def process_candle(ledger, candle):
    """Update open positions using one confirmed 5M OHLC candle."""
    timestamp = candle["timestamp"] + FIVE_MINUTES_SECONDS
    high = _number(candle.get("high"), "candle high")
    low = _number(candle.get("low"), "candle low")
    close = _number(candle.get("close"), "candle close")

    for trade in ledger["trades"]:
        if trade.get("status") != "OPEN" or timestamp <= trade["entry_timestamp"]:
            continue
        direction = trade["direction"]
        stop = trade["stop_loss_current"]
        tp1 = trade["take_profit_1"]

        # If stop and TP1 are both touched in one OHLC candle, assume stop first.
        stop_hit = low <= stop if direction == "LONG" else high >= stop
        tp1_hit = high >= tp1 if direction == "LONG" else low <= tp1
        if stop_hit:
            reason = "TRAILING_STOP" if trade["trailing_active"] else "STOP_LOSS"
            _close_trade(trade, stop, timestamp, reason)
            continue
        if tp1_hit:
            _close_trade(trade, tp1, timestamp, "TAKE_PROFIT_1_FULL_EXIT")
            continue

        if direction == "LONG":
            trade["tp2_reached"] = trade["tp2_reached"] or high >= trade["take_profit_2"]
            trade["tp3_reached"] = trade["tp3_reached"] or high >= trade["take_profit_3"]
            favorable_price = high
        else:
            trade["tp2_reached"] = trade["tp2_reached"] or low <= trade["take_profit_2"]
            trade["tp3_reached"] = trade["tp3_reached"] or low <= trade["take_profit_3"]
            favorable_price = low

        favorable_net = _net_at_exit(trade, favorable_price)
        if not trade["trailing_active"] and favorable_net >= TRAIL_TRIGGER_NET_USD:
            trade["trailing_active"] = True
            trade["stop_loss_current"] = price_for_net_profit(
                trade, TRAIL_LOCK_NET_USD
            )
            trade["trailing_activated_timestamp"] = timestamp
            trade["trailing_activated_time_utc"] = utc_iso(timestamp)


        # Intrabar order is unknowable; if the newly established trailing stop
        # was also crossed, conservatively record the locked-profit stop fill.
        if trade["trailing_active"]:
            trailing_stop = trade["stop_loss_current"]
            trailing_hit = (
                low <= trailing_stop if direction == "LONG"
                else high >= trailing_stop
            )
            if trailing_hit:
                _close_trade(trade, trailing_stop, timestamp, "TRAILING_STOP")
                continue

        mark = _net_at_exit(trade, close)
        trade["unrealized_pnl_usd"] = mark
        trade["net_pnl_usd"] = mark
        trade["last_mark_price"] = close
        trade["last_updated_at"] = utc_iso(timestamp)


def process_candles(ledger, candles):
    for candle in candles:
        process_candle(ledger, candle)
    return ledger


def summary(ledger):
    trades = ledger["trades"]
    closed = [trade for trade in trades if trade.get("status") == "CLOSED"]
    opened = [trade for trade in trades if trade.get("status") == "OPEN"]
    realized = sum(trade.get("realized_pnl_usd", 0.0) for trade in closed)
    unrealized = sum(trade.get("unrealized_pnl_usd", 0.0) for trade in opened)
    wins = sum(1 for trade in closed if trade.get("realized_pnl_usd", 0.0) > 0)
    return {
        "total_trades": len(trades),
        "open_trades": len(opened),
        "closed_trades": len(closed),
        "winning_closed_trades": wins,
        "win_rate_percent": (wins / len(closed) * 100) if closed else 0.0,
        "realized_net_pnl_usd": realized,
        "unrealized_net_pnl_usd": unrealized,
        "combined_net_pnl_usd": realized + unrealized,
    }
