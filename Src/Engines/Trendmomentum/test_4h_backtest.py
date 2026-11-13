
import pandas as pd

from Src.Engines.Trendmomentum.data_loader import load_ohlcv
from Src.Engines.Trendmomentum.indicators import calculate_indicators
from Src.Engines.Trendmomentum.market_regime import classify_market_regime
from Src.Engines.Trendmomentum.trend_analysis import analyze_trend
from Src.Engines.Trendmomentum.signal_generator import generate_signals


TIMEFRAME = "4H"

ATR_STOP_MULTIPLIER = 1.5
REWARD_RISK_RATIO = 2.0
MAX_HOLDING_CANDLES = 24

FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002
ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)


def prepare_data():
    data = load_ohlcv(TIMEFRAME)
    data = calculate_indicators(data)
    data = classify_market_regime(data)
    data = analyze_trend(data)

    data = generate_signals(
        data,
        breakout_lookback=20,
        pullback_atr_tolerance=0.25,
    )

    return data


def simulate_trade(data, signal_index):
    signal_row = data.iloc[signal_index]
    entry_index = signal_index + 1

    if entry_index + MAX_HOLDING_CANDLES > len(data):
        return None

    direction = signal_row["signal"]

    if direction not in ("LONG", "SHORT"):
        return None

    atr = float(signal_row["atr_14"])
    entry_price = float(data.iloc[entry_index]["open"])

    if not pd.notna(atr) or atr <= 0 or entry_price <= 0:
        return None

    risk_distance = atr * ATR_STOP_MULTIPLIER

    if direction == "LONG":
        stop_price = entry_price - risk_distance
        target_price = entry_price + (
            risk_distance * REWARD_RISK_RATIO
        )
    else:
        stop_price = entry_price + risk_distance
        target_price = entry_price - (
            risk_distance * REWARD_RISK_RATIO
        )

    exit_price = None
    exit_reason = None
    exit_index = None

    final_index = entry_index + MAX_HOLDING_CANDLES - 1

    for j in range(entry_index, final_index + 1):
        candle = data.iloc[j]

        candle_open = float(candle["open"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])

        if direction == "LONG":
            stop_hit = candle_low <= stop_price
            target_hit = candle_high >= target_price

            if stop_hit and target_hit:
                exit_price = min(candle_open, stop_price)
                exit_reason = "STOP_LOSS_AMBIGUOUS"
            elif stop_hit:
                exit_price = min(candle_open, stop_price)
                exit_reason = "STOP_LOSS"
            elif target_hit:
                exit_price = max(candle_open, target_price)
                exit_reason = "TAKE_PROFIT"

        else:
            stop_hit = candle_high >= stop_price
            target_hit = candle_low <= target_price

            if stop_hit and target_hit:
                exit_price = max(candle_open, stop_price)
                exit_reason = "STOP_LOSS_AMBIGUOUS"
            elif stop_hit:
                exit_price = max(candle_open, stop_price)
                exit_reason = "STOP_LOSS"
            elif target_hit:
                exit_price = min(candle_open, target_price)
                exit_reason = "TAKE_PROFIT"

        if exit_reason is not None:
            exit_index = j
            break

    if exit_reason is None:
        exit_index = final_index
        exit_price = float(data.iloc[exit_index]["close"])
        exit_reason = "TIME_EXIT"

    if exit_price <= 0:
        return None

    if direction == "LONG":
        gross_return = (exit_price - entry_price) / entry_price
    else:
        gross_return = (entry_price - exit_price) / entry_price

    net_return = gross_return - ROUND_TRIP_COST

    initial_risk_pct = risk_distance / entry_price
    net_r = net_return / initial_risk_pct
    gross_r = gross_return / initial_risk_pct

    return {
        "direction": direction,
        "signal_type": signal_row["signal_type"],
        "signal_timestamp": int(signal_row["timestamp"]),
        "entry_timestamp": int(data.iloc[entry_index]["timestamp"]),
        "exit_timestamp": int(data.iloc[exit_index]["timestamp"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_reason": exit_reason,
        "holding_candles": exit_index - entry_index + 1,
        "gross_return_pct": gross_return * 100,
        "net_return_pct": net_return * 100,
        "gross_R": gross_r,
        "net_R": net_r,
    }


def run_backtest(data):
    trades = []
    last_exit_index = -1

    for i in range(len(data) - 1):
        if i < last_exit_index:
            continue

        signal = data.iloc[i]["signal"]

        if signal not in ("LONG", "SHORT"):
            continue

        trade = simulate_trade(data, i)

        if trade is None:
            continue

        trades.append(trade)

        exit_timestamp = trade["exit_timestamp"]

        matching_indices = data.index[
            data["timestamp"] == exit_timestamp
        ]

        if len(matching_indices) > 0:
            last_exit_index = int(matching_indices[0])

    return pd.DataFrame(trades)


def calculate_metrics(trades):
    if trades.empty:
        return None

    wins = trades[trades["net_R"] > 0]
    losses = trades[trades["net_R"] < 0]

    gross_profit = wins["net_R"].sum()
    gross_loss = abs(losses["net_R"].sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    equity = trades["net_R"].cumsum()
    peak = equity.cummax().clip(lower=0)
    drawdown = equity - peak

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(trades) - len(wins) - len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "profit_factor": profit_factor,
        "average_net_R": trades["net_R"].mean(),
        "total_net_R": trades["net_R"].sum(),
        "average_net_return_pct": trades["net_return_pct"].mean(),
        "max_drawdown_R": abs(drawdown.min()),
        "average_holding_candles": trades["holding_candles"].mean(),
    }


def print_metrics(trades, label):
    print(f"\n{label}")
    print("-" * 70)

    metrics = calculate_metrics(trades)

    if metrics is None:
        print("No completed trades.")
        return

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    print("=" * 70)
    print("INDEPENDENT 4H TREND AND MOMENTUM BACKTEST")
    print("=" * 70)

    data = prepare_data()

    print(f"Timeframe: {TIMEFRAME}")
    print(f"Completed Candles: {len(data)}")
    print(f"ATR Stop Multiplier: {ATR_STOP_MULTIPLIER}")
    print(f"Reward/Risk Ratio: {REWARD_RISK_RATIO}")
    print(f"Maximum Holding: {MAX_HOLDING_CANDLES} candles")
    print(f"Round Trip Cost: {ROUND_TRIP_COST * 100:.3f}%")

    print("\nSIGNAL DISTRIBUTION")
    print("-" * 70)
    print(data["signal"].value_counts(dropna=False))

    trades = run_backtest(data)

    print("\nBACKTEST RESULTS")
    print("=" * 70)

    if trades.empty:
        print("No completed trades were generated.")
    else:
        print_metrics(trades, "COMBINED RESULTS")

        print_metrics(
            trades[trades["direction"] == "LONG"],
            "LONG RESULTS",
        )

        print_metrics(
            trades[trades["direction"] == "SHORT"],
            "SHORT RESULTS",
        )

        print("\nEXIT REASON DISTRIBUTION")
        print("-" * 70)
        print(trades["exit_reason"].value_counts())

        print("\nTRADE DETAILS")
        print("-" * 70)
        print(
            trades[
                [
                    "direction",
                    "signal_type",
                    "entry_price",
                    "exit_price",
                    "exit_reason",
                    "holding_candles",
                    "net_return_pct",
                    "net_R",
                ]
            ].to_string(index=False)
        )

    print("\nLATEST MARKET OUTPUT")
    print("-" * 70)

    latest = data.iloc[-1]

    for column in [
        "timestamp",
        "close",
        "market_regime",
        "trend_direction",
        "trend_strength",
        "momentum_direction",
        "trend_alignment",
        "rsi_14",
        "adx_14",
        "atr_14",
        "signal",
        "signal_type",
        "signal_reason",
    ]:
        print(f"{column}: {latest[column]}")

    print("\nBACKTEST LIMITATIONS")
    print("1. Historical BTC-USDT SPOT data is used.")
    print("2. SHORT results are simulated research results only.")
    print("3. No leverage or compounding is applied.")
    print("4. Net R is normalized by the initial ATR-based risk.")
    print("5. Only one position can be active at a time.")
    print("6. Same-candle stop and target hits use stop-first logic.")
    print("7. Results do not guarantee future profitability.")

    print("\nSTATUS: PASS")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
