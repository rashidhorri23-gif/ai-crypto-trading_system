
import math


def calculate_trade_plan(
    signal,
    entry_price,
    atr,
    account_equity,
    risk_percent=1.0,
    atr_multiplier=1.5,
    reward_risk_ratio=2.0,
    fee_rate=0.0005,
    slippage_rate=0.0002,
    max_position_percent=100.0
):
    if signal not in ("LONG", "SHORT"):
        raise ValueError("Signal must be LONG or SHORT.")

    values = {
        "entry_price": entry_price,
        "atr": atr,
        "account_equity": account_equity,
        "risk_percent": risk_percent,
        "atr_multiplier": atr_multiplier,
        "reward_risk_ratio": reward_risk_ratio,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "max_position_percent": max_position_percent
    }

    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if entry_price <= 0 or atr <= 0 or account_equity <= 0:
        raise ValueError("Price, ATR, and equity must be positive.")

    if not 0 < risk_percent <= 100:
        raise ValueError("Risk percent must be between 0 and 100.")

    if atr_multiplier <= 0 or reward_risk_ratio <= 0:
        raise ValueError("ATR multiplier and reward-risk ratio must be positive.")

    if not 0 <= fee_rate < 1:
        raise ValueError("Fee rate must be between 0 and 1.")

    if not 0 <= slippage_rate < 1:
        raise ValueError("Slippage rate must be between 0 and 1.")

    if not 0 < max_position_percent <= 100:
        raise ValueError("Maximum position percent must be between 0 and 100.")

    stop_distance = atr * atr_multiplier

    if signal == "LONG":
        actual_entry = entry_price * (1 + slippage_rate)
        stop_loss = actual_entry - stop_distance
        take_profit = actual_entry + stop_distance * reward_risk_ratio

        stop_exit = stop_loss * (1 - slippage_rate)
        target_exit = take_profit * (1 - slippage_rate)

    else:
        actual_entry = entry_price * (1 - slippage_rate)
        stop_loss = actual_entry + stop_distance
        take_profit = actual_entry - stop_distance * reward_risk_ratio

        stop_exit = stop_loss * (1 + slippage_rate)
        target_exit = take_profit * (1 + slippage_rate)

    if min(actual_entry, stop_loss, take_profit, stop_exit, target_exit) <= 0:
        raise ValueError("Calculated execution prices must be positive.")

    entry_fee_per_unit = actual_entry * fee_rate
    stop_exit_fee_per_unit = stop_exit * fee_rate

    risk_per_unit = (
        abs(actual_entry - stop_exit)
        + entry_fee_per_unit
        + stop_exit_fee_per_unit
    )

    reward_per_unit = (
        abs(target_exit - actual_entry)
        - entry_fee_per_unit
        - target_exit * fee_rate
    )

    if risk_per_unit <= 0:
        raise ValueError("Calculated risk per unit must be positive.")

    if reward_per_unit <= 0:
        raise ValueError("Estimated net reward is not positive.")

    risk_amount = account_equity * risk_percent / 100

    max_position_value = (
        account_equity * max_position_percent / 100
    )

    max_position_size = max_position_value / actual_entry

    risk_based_position_size = risk_amount / risk_per_unit

    position_size = min(
        risk_based_position_size,
        max_position_size
    )

    position_value = position_size * actual_entry

    estimated_risk = position_size * risk_per_unit
    estimated_net_reward = position_size * reward_per_unit

    estimated_net_rr = estimated_net_reward / estimated_risk

    return {
        "signal": signal,
        "reference_entry_price": entry_price,
        "entry_price": actual_entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_exit_price_estimate": stop_exit,
        "target_exit_price_estimate": target_exit,
        "stop_distance": stop_distance,
        "risk_amount": risk_amount,
        "estimated_risk": estimated_risk,
        "risk_percent": risk_percent,
        "position_size": position_size,
        "position_value": position_value,
        "max_position_value": max_position_value,
        "reward_risk_ratio": reward_risk_ratio,
        "estimated_net_reward": estimated_net_reward,
        "estimated_net_rr": estimated_net_rr,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "atr_multiplier": atr_multiplier
    }
