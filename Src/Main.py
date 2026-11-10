
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings
from Src.Data.okx_client import fetch_historical_candles


def main():
    print("AI Crypto Trading System")
    print("========================")
    print("Historical Data Test")
    print(f"Provider: {settings.DATA_PROVIDER}")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Timeframe: 5m")
    print("Requested Candles: 1000")
    print("Live Trading: DISABLED")

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError("Live trading must remain disabled.")

    candles = fetch_historical_candles(
        market="SPOT",
        timeframe="5m",
        total_candles=1000,
        page_size=100
    )

    if not candles:
        raise RuntimeError("No historical candles returned.")

    timestamps = [
        candle["timestamp"]
        for candle in candles
    ]

    unique_timestamps = set(timestamps)

    duplicates = len(timestamps) - len(unique_timestamps)

    interval_ms = 5 * 60 * 1000
    missing_intervals = []

    for index in range(1, len(candles)):
        difference = (
            candles[index]["timestamp"]
            - candles[index - 1]["timestamp"]
        )

        if difference > interval_ms:
            missing_intervals.append(
                (difference // interval_ms) - 1
            )

    oldest = datetime.fromtimestamp(
        candles[0]["timestamp"] / 1000,
        tz=timezone.utc
    )

    newest = datetime.fromtimestamp(
        candles[-1]["timestamp"] / 1000,
        tz=timezone.utc
    )

    print("\nHistorical Data Results")
    print("=======================")
    print(f"Returned Candles: {len(candles)}")
    print(f"Unique Candles: {len(unique_timestamps)}")
    print(f"Duplicate Candles: {duplicates}")
    print(f"Missing Intervals: {sum(missing_intervals)}")
    print(f"Oldest UTC: {oldest.isoformat()}")
    print(f"Newest UTC: {newest.isoformat()}")
    print(f"First Close: {candles[0]['close']}")
    print(f"Last Close: {candles[-1]['close']}")

    if len(candles) < 1000:
        print("\nWARNING: Fewer than 1000 candles returned.")

    if duplicates:
        raise RuntimeError("Duplicate timestamps detected.")

    if missing_intervals:
        print("\nWARNING: Historical gaps detected.")
    else:
        print("\nNo missing 5m intervals detected.")

    print("\nHISTORICAL TEST COMPLETED")
    print("LIVE TRADING: DISABLED")


if __name__ == "__main__":
    main()
