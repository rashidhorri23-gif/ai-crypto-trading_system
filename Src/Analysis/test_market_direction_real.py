
from Src.Analysis.daily_structure import analyze_daily_structure
from Src.Analysis.market_direction import determine_market_direction

from Src.Analysis.test_daily_structure_real import (
    load_candles as load_daily_candles
)

from Src.Analysis.test_4h_structure_real import (
    load_candles as load_four_hour_candles
)


def run_market_direction_test():
    print("=" * 65)
    print("REAL BTC DAILY + 4H MARKET DIRECTION TEST")
    print("=" * 65)

    daily_candles = load_daily_candles()
    four_hour_candles = load_four_hour_candles()

    print("Daily candles:", len(daily_candles))
    print("4H candles:", len(four_hour_candles))

    if len(daily_candles) < 20:
        raise ValueError("Insufficient daily candles.")

    if len(four_hour_candles) < 20:
        raise ValueError("Insufficient 4H candles.")

    daily_result = analyze_daily_structure(daily_candles)
    four_hour_result = analyze_daily_structure(four_hour_candles)

    decision = determine_market_direction(
        daily_result,
        four_hour_result
    )

    print("\n" + "=" * 65)
    print("DAILY STRUCTURE")
    print("=" * 65)
    print("Trend:", decision["daily_bias"])
    print("Event Bias:", decision["daily_event_bias"])
    print("Reason:", decision["daily_structure_reason"])

    print("\n" + "=" * 65)
    print("4H STRUCTURE")
    print("=" * 65)
    print("Trend:", decision["four_hour_bias"])
    print("Event Bias:", decision["four_hour_event_bias"])
    print("Reason:", decision["four_hour_structure_reason"])

    print("\n" + "=" * 65)
    print("FINAL MARKET DIRECTION")
    print("=" * 65)
    print("Allowed Direction:", decision["allowed_direction"])
    print("Status:", decision["status"])
    print("Reason:", decision["reason"])
    print(
        "Trade Directions:",
        decision["trade_directions_to_evaluate"]
    )

    print("\n" + "=" * 65)
    print("MARKET DIRECTION TEST COMPLETED")
    print("=" * 65)

    return decision


if __name__ == "__main__":
    run_market_direction_test()
