
import numpy as np

from Src.Engines.Trendmomentum.data_loader import load_all_timeframes
from Src.Engines.Trendmomentum.indicators import calculate_indicators


def main():
    print("TREND & MOMENTUM INDICATOR TEST")
    print("================================")

    datasets = load_all_timeframes()

    all_passed = True

    for timeframe, data in datasets.items():
        print(f"\nTIMEFRAME: {timeframe}")
        print("-" * 40)

        if len(data) < 250:
            print("FAIL: Not enough candles for EMA200.")
            all_passed = False
            continue

        result = calculate_indicators(data)

        required = [
            "ema_20",
            "ema_50",
            "ema_200",
            "rsi_14",
            "atr_14",
            "plus_di_14",
            "minus_di_14",
            "adx_14",
            "ema_200_slope"
        ]

        latest = result.iloc[-1]

        invalid = [
            column
            for column in required
            if not np.isfinite(latest[column])
        ]

        if invalid:
            print(f"FAIL: Invalid latest indicators: {invalid}")
            all_passed = False
            continue

        print(f"Total Candles: {len(result)}")
        print(f"First Timestamp: {result.iloc[0]['timestamp']}")
        print(f"Last Timestamp: {latest['timestamp']}")
        print(f"Latest Close: {latest['close']:.2f}")
        print(f"EMA 20: {latest['ema_20']:.2f}")
        print(f"EMA 50: {latest['ema_50']:.2f}")
        print(f"EMA 200: {latest['ema_200']:.2f}")
        print(f"RSI 14: {latest['rsi_14']:.2f}")
        print(f"ATR 14: {latest['atr_14']:.2f}")
        print(f"ADX 14: {latest['adx_14']:.2f}")
        print(f"Plus DI: {latest['plus_di_14']:.2f}")
        print(f"Minus DI: {latest['minus_di_14']:.2f}")
        print(f"EMA 200 Slope: {latest['ema_200_slope']:.2f}")

        print("STATUS: PASS")

    print("\nFINAL INDICATOR TEST")
    print("====================")
    print(f"RESULT: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        raise RuntimeError("Indicator test failed.")


if __name__ == "__main__":
    main()
