BTC Paper Trading

This module records simulated BTC-USDT trades only. It never submits exchange orders.

Rules

- A paper trade is created only after the existing 4H activation and confirmed 5M retest logic succeeds.
- Each trade uses $1,000 simulated margin and 3x leverage, for $3,000 notional exposure.
- The entry reference is the confirmed 5M retest candle close. Entry and exit slippage are applied adversely at 0.02% per side.
- Trading fees are 0.05% per side. Net PnL includes both entry and exit fees plus slippage through simulated fill prices.
- The entire position closes at TP1. TP2 and TP3 remain stored as plan levels; there are no partial exits.
- When net unrealized PnL reaches $70, the stop is moved to a price that locks at least $50 net profit after fees and slippage.
- The trailing giveback distance is intentionally unset. After activation, the stop remains at the level that locks at least $50 net profit; any further trailing adjustment is disabled until the user chooses a giveback rule.
- If one 5M candle touches both the active stop and TP1, stop-first is assumed. This is the conservative OHLC assumption because intrabar tick order is unavailable.
- Multiple paper trades may coexist. Each trade has its own independent $1,000 simulated margin; the ledger is not constrained to a $1,000 aggregate account balance.

Files

- "paper_trading.json": persistent trade ledger and performance summary.
- "Src/Engines/AutonomousTechnicalAnalyst/paper_trading.py": accounting and position-management engine.
- The existing 5M monitor creates trades after retest confirmation and updates open positions from newly confirmed 5M candles.

Pending user decision

The trailing giveback distance has deliberately been left unset ("TRAIL_GIVEBACK_NET_USD = None"). The $70 activation and $50 minimum net-profit lock remain active. After activation, the stop stays at the initial $50 net-profit lock; it does not trail further until the user chooses a giveback rule. This avoids silently imposing an unapproved distance.

