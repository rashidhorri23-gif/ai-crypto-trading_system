
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings
from Src.Data.okx_client import fetch_candles


def main():
    print("AI Crypto Trading System")
    print("========================")
    print(f"Data Provider: {settings.DATA_PROVIDER}")
    print("Execution Mode: MANUAL")
    print("Live Trading: DISABLED")

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError("Live trading must remain disabled.")

    markets = ["SPOT", "SWAP"]
    timeframes = ["5m", "15m", "1H", "4H", "1D"]

    failures = []

    for market in markets:
        for timeframe in timeframes:
            print(f"\nTesting {market} {timeframe}")

            try:
                candles = fetch_candles(
                    market,
                    timeframe,
                    limit=100
                )

                if not candles:
                    raise RuntimeError("No completed candles returned.")

                latest = candles[-1]

                print(f"Completed Candles: {len(candles)}")
                print(f"Latest UTC Timestamp: {latest['timestamp']}")
                print(f"Latest Close: {latest['close']}")
                print("Result: SUCCESS")

            except Exception as error:
                print(f"Result: FAILED")
                print(f"Error: {error}")
                failures.append(f"{market} {timeframe}")

    print("\n========================")

    if failures:
        raise RuntimeError(
            "Failed tests: " + ", ".join(failures)
        )

    print("ALL CANDLE TESTS PASSED")
    print("SPOT: SUCCESS")
    print("SWAP: SUCCESS")
    print("LIVE TRADING: DISABLED")


if __name__ == "__main__":
    main()
