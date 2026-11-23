
"""BTC-only paper trading engine. No exchange orders are submitted."""

import hashlib
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
TRAIL_GIVEBACK_NET_USD = None
FIVE_MINUTES_SECONDS = 300


def utc_iso(timestamp=None):
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat()

    return datetime.fromtimestamp(
        timestamp,
        timezone.utc,
    ).isoformat()


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


def load_ledger(path=None):
    path = path or TRADING_FILE

    if not path.exists():
        return {
            "version": 1,
            "settings": settings(),
            "trades": [],
        }

    with path.open("r", encoding="utf-8") as handle:
        ledger = json.load(handle)

    if (
        not isinstance(ledger, dict)
        or not isinstance(ledger.get("trades"), list)
    ):
        raise ValueError(
            "Paper trading ledger has an invalid structure."
        )

    ledger.setdefault("version", 1)
    ledger.setdefault("settings", settings())

    return ledger


def save_ledger(ledger, path=None):
    path = path or TRADING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            ledger,
            handle,
            indent=2,
            allow_nan=False,
        )

    temporary.replace(path)


def _number(value, field):
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(f"{field} must be numeric.")

    value = float(value)

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{field} must be a positive finite number."
        )

    return value


def _timestamp(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a numeric timestamp.")

    value = int(value)

    if value <= 0:
        raise ValueError(f"{field} must be positive.")

    return value


def _fills(direction, entry_price, exit_price):
    if direction == "LONG":
        return (
            entry_price * (1 + SLIPPAGE_RATE),
            exit_price * (1 - SLIPPAGE_RATE),
        )

    if direction == "SHORT":
        return (
            entry_price * (1 - SLIPPAGE_RATE),
            exit_price * (1 + SLIPPAGE_RATE),
        )

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
        trade["direction"],
        trade["quantity"],
        trade["entry_fill_price"],
        exit_price,
    )["net_pnl_usd"]


def price_for_net_profit(trade, target_net):
    """Calculate the exit reference price for a target net PnL."""

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


def _validate_trade_levels(direction, entry_price, stop, tp1, tp2, tp3):
    if direction == "LONG":
        if not stop < entry_price:
            raise ValueError("LONG stop loss must be below entry.")

        if not entry_price < tp1 <= tp2 <= tp3:
            raise ValueError("Invalid LONG target ordering.")

    elif direction == "SHORT":
        if not stop > entry_price:
            raise ValueError("SHORT stop loss must be above entry.")

        if not entry_price > tp1 >= tp2 >= tp3:
            raise ValueError("Invalid SHORT target ordering.")

    else:
        raise ValueError("Direction must be LONG or SHORT.")


def make_signal_fingerprint(analysis_id, direction, entry_type, entry_price):
    raw = "|".join([
        str(analysis_id),
        str(direction),
        str(entry_type),
        str(entry_price),
    ])

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def open_trade_from_ai(
    ledger,
    direction,
    entry_type,
    entry_price,
    stop_loss,
    take_profit_1,
    take_profit_2,
    take_profit_3,
    signal_fingerprint,
    timestamp=None,
):
    """Open a paper trade using the AI-selected entry price."""

    if direction not in ("LONG", "SHORT"):
        raise ValueError("Direction must be LONG or SHORT.")

    if entry_type not in ("MARKET", "LIMIT"):
        raise ValueError("Entry type must be MARKET or LIMIT.")

    if not isinstance(signal_fingerprint, str) or not signal_fingerprint:
        raise ValueError("Signal fingerprint is required.")

    entry_reference = _number(entry_price, "entry_price")
    stop = _number(stop_loss, "stop_loss")
    tp1 = _number(take_profit_1, "take_profit_1")
    tp2 = _number(take_profit_2, "take_profit_2")
    tp3 = _number(take_profit_3, "take_profit_3")

    _validate_trade_levels(
        direction,
        entry_reference,
        stop,
        tp1,
        tp2,
        tp3,
    )

    for existing in ledger["trades"]:
        if existing.get("signal_fingerprint") == signal_fingerprint:
            return existing, False

    timestamp = _timestamp(
        timestamp if timestamp is not None else datetime.now(
            timezone.utc
        ).timestamp(),
        "entry timestamp",
    )

    entry_fill, _ = _fills(
        direction,
        entry_reference,
        entry_reference,
    )

    quantity = NOTIONAL_USD / entry_fill
    entry_fee = quantity * entry_fill * FEE_RATE

    trade = {
        "trade_id": signal_fingerprint,
        "signal_fingerprint": signal_fingerprint,
        "symbol": "BTC-USDT",
        "direction": direction,
        "entry_type": entry_type,
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
        "last_mark_price": entry_reference,
        "last_updated_at": utc_iso(timestamp),
    }

    ledger["trades"].append(trade)

    return trade, True


def open_trade(
    ledger,
    direction,
    plan,
    confirmation_candle,
    signal_fingerprint,
):
    """Legacy-compatible entry function."""

    if not isinstance(confirmation_candle, dict):
        raise ValueError("Confirmation candle is required.")

    return open_trade_from_ai(
        ledger=ledger,
        direction=direction,
        entry_type="MARKET",
        entry_price=confirmation_candle.get("close"),
        stop_loss=plan.get("stop_loss"),
        take_profit_1=plan.get("take_profit_1"),
        take_profit_2=plan.get("take_profit_2"),
        take_profit_3=plan.get("take_profit_3"),
        signal_fingerprint=signal_fingerprint,
        timestamp=(
            confirmation_candle["timestamp"]
            + FIVE_MINUTES_SECONDS
        ),
    )


def get_trade(ledger, trade_id):
    matches = [
        trade
        for trade in ledger["trades"]
        if trade.get("trade_id") == trade_id
    ]

    if not matches:
        raise ValueError(f"Trade not found: {trade_id}")

    if len(matches) != 1:
        raise ValueError(f"Duplicate trade ID detected: {trade_id}")

    return matches[0]


def update_trade_levels(
    trade,
    stop_loss=None,
    take_profit_1=None,
    take_profit_2=None,
    take_profit_3=None,
):
    """Apply AI-approved level changes without weakening a locked stop."""

    if trade.get("status") != "OPEN":
        raise ValueError("Cannot update a closed trade.")

    entry = _number(
        trade["entry_reference_price"],
        "entry_reference_price",
    )

    direction = trade["direction"]

    new_stop = (
        trade["stop_loss_current"]
        if stop_loss is None
        else _number(stop_loss, "stop_loss")
    )

    new_tp1 = (
        trade["take_profit_1"]
        if take_profit_1 is None
        else _number(take_profit_1, "take_profit_1")
    )

    new_tp2 = (
        trade["take_profit_2"]
        if take_profit_2 is None
        else _number(take_profit_2, "take_profit_2")
    )

    new_tp3 = (
        trade["take_profit_3"]
        if take_profit_3 is None
        else _number(take_profit_3, "take_profit_3")
    )

    _validate_trade_levels(
        direction,
        entry,
        new_stop,
        new_tp1,
        new_tp2,
        new_tp3,
    )

    if trade.get("trailing_active"):
        old_stop = trade["stop_loss_current"]

        if direction == "LONG" and new_stop < old_stop:
            raise ValueError(
                "AI update cannot weaken an active LONG trailing stop."
            )

        if direction == "SHORT" and new_stop > old_stop:
            raise ValueError(
                "AI update cannot weaken an active SHORT trailing stop."
            )

    trade["stop_loss_current"] = new_stop
    trade["take_profit_1"] = new_tp1
    trade["take_profit_2"] = new_tp2
    trade["take_profit_3"] = new_tp3
    trade["last_updated_at"] = utc_iso()
    trade["levels_updated_by"] = "AI"

    return trade


def _close_trade(trade, exit_price, timestamp, reason):
    result = trade_pnl(
        trade["direction"],
        trade["quantity"],
        trade["entry_fill_price"],
        exit_price,
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
        "last_updated_at": utc_iso(timestamp),
    })


def close_trade_at_market(trade, reference_price, timestamp, reason="AI_CLOSE"):
    """Close an open paper trade at a supplied reference price."""

    if trade.get("status") != "OPEN":
        raise ValueError("Trade is not open.")

    price = _number(reference_price, "reference_price")
    timestamp = _timestamp(timestamp, "exit timestamp")

    _close_trade(
        trade,
        price,
        timestamp,
        reason,
    )

    return trade


def process_candle(ledger, candle):
    """Manage open positions using one confirmed 5M OHLC candle."""

    timestamp = _timestamp(
        candle["timestamp"],
        "candle timestamp",
    ) + FIVE_MINUTES_SECONDS

    high = _number(candle.get("high"), "candle high")
    low = _number(candle.get("low"), "candle low")
    close = _number(candle.get("close"), "candle close")

    if high < low:
        raise ValueError("Invalid candle high/low geometry.")

    for trade in ledger["trades"]:
        if trade.get("status") != "OPEN":
            continue

        if timestamp <= trade["entry_timestamp"]:
            continue

        direction = trade["direction"]
        stop = trade["stop_loss_current"]
        tp1 = trade["take_profit_1"]

        # Conservative assumption: stop first if both levels are touched.
        stop_hit = (
            low <= stop
            if direction == "LONG"
            else high >= stop
        )

        tp1_hit = (
            high >= tp1
            if direction == "LONG"
            else low <= tp1
        )

        if stop_hit:
            reason = (
                "TRAILING_STOP"
                if trade["trailing_active"]
                else "STOP_LOSS"
            )

            _close_trade(
                trade,
                stop,
                timestamp,
                reason,
            )
            continue

        if tp1_hit:
            _close_trade(
                trade,
                tp1,
                timestamp,
                "TAKE_PROFIT_1_FULL_EXIT",
            )
            continue

        if direction == "LONG":
            trade["tp2_reached"] = (
                trade["tp2_reached"]
                or high >= trade["take_profit_2"]
            )

            trade["tp3_reached"] = (
                trade["tp3_reached"]
                or high >= trade["take_profit_3"]
            )

            favorable_price = high

        else:
            trade["tp2_reached"] = (
                trade["tp2_reached"]
                or low <= trade["take_profit_2"]
            )

            trade["tp3_reached"] = (
                trade["tp3_reached"]
                or low <= trade["take_profit_3"]
            )

            favorable_price = low

        favorable_net = _net_at_exit(
            trade,
            favorable_price,
        )

        if (
            not trade["trailing_active"]
            and favorable_net >= TRAIL_TRIGGER_NET_USD
        ):
            trade["trailing_active"] = True

            trade["stop_loss_current"] = price_for_net_profit(
                trade,
                TRAIL_LOCK_NET_USD,
            )

            trade["trailing_activated_timestamp"] = timestamp
            trade["trailing_activated_time_utc"] = utc_iso(timestamp)

        # Conservatively process a newly crossed trailing stop.
        if trade["trailing_active"]:
            trailing_stop = trade["stop_loss_current"]

            trailing_hit = (
                low <= trailing_stop
                if direction == "LONG"
                else high >= trailing_stop
            )

            if trailing_hit:
                _close_trade(
                    trade,
                    trailing_stop,
                    timestamp,
                    "TRAILING_STOP",
                )
                continue

        mark = _net_at_exit(
            trade,
            close,
        )

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

    closed = [
        trade
        for trade in trades
        if trade.get("status") == "CLOSED"
    ]

    opened = [
        trade
        for trade in trades
        if trade.get("status") == "OPEN"
    ]

    realized = sum(
        trade.get("realized_pnl_usd", 0.0)
        for trade in closed
    )

    unrealized = sum(
        trade.get("unrealized_pnl_usd", 0.0)
        for trade in opened
    )

    wins = sum(
        1
        for trade in closed
        if trade.get("realized_pnl_usd", 0.0) > 0
    )

    return {
        "total_trades": len(trades),
        "open_trades": len(opened),
        "closed_trades": len(closed),
        "winning_closed_trades": wins,
        "win_rate_percent": (
            wins / len(closed) * 100
            if closed
            else 0.0
        ),
        "realized_net_pnl_usd": realized,
        "unrealized_net_pnl_usd": unrealized,
        "combined_net_pnl_usd": realized + unrealized,
    }
