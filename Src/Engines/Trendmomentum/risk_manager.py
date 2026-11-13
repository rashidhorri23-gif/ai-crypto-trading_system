
import math


def calculate_trade_plan(
    signal,
    entry_price,
    atr,
    account_equity,
    risk_percent=1.0,
    atr_multiplier=1.5,
    reward_risk_ratio=2.0
):
    if signal not in ("LONG", "SHORT"):
        raise ValueError("Signal must be LONG or SHORT.")

    values = {
        "entry_price": entry_price,
        "atr": atr,
        "account_equity": account_equity,
        "risk_percent": risk_percent,
        "atr_multiplier": atr_multiplier,
        "reward_risk_ratio": reward_risk_ratio
    }

    for name, value in values.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if entry_price <= 0:
        raise ValueError("Entry price must be positive.")

    if atr <= 0:
        raise ValueError("ATR must be positive.")

    if account_equity <= 0:
        raise ValueError("Account equity must be positive.")

    if not 0 < risk_percent <= 100:
        raise ValueError("Risk percent must be between 0 and 100.")

    if atr_multiplier <= 0 or reward_risk_ratio <= 0:
        raise ValueError("Risk parameters must be positive.")

    stop_distance = atr * atr_multiplier
    risk_amount = account_equity * risk_percent / 100

    if signal == "LONG":
        stop_loss = entry_price - stop_distance
        take_profit = (
            entry_price
            + stop_distance * reward_risk_ratio
        )
    else:
        stop_loss = entry_price + stop_distance
        take_profit = (
            entry_price
            - stop_distance * reward_risk_ratio
        )

    if stop_loss <= 0 or take_profit <= 0:
        raise ValueError("Calculated price levels must be positive.")

    position_size = risk_amount / stop_distance
    position_value = position_size * entry_price

    return {
        "signal": signal,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance": stop_distance,
        "risk_amount": risk_amount,
        "risk_percent": risk_percent,
        "position_size": position_size,
        "position_value": position_value,
        "reward_risk_ratio": reward_risk_ratio,
        "atr_multiplier": atr_multiplier
    }
