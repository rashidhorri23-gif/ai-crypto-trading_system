
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings
from Src.Data.okx_client import fetch_new_candles
from Src.Data.archive import save_candles, get_latest_timestamp


def main():
    print("AI Crypto Trading System")
    print("========================")
    print(f"Provider: {settings.DATA_PROVIDER}")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Timeframe: 5m")
    print("Mode: INCREMENTAL ARCHIVE")
    print("Live Trading: DISABLED")

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError("Live trading must remain disabled.")

    filename = "BTC_USDT_SPOT_5m.csv"

    last_timestamp = get_latest_timestamp(filename)

    if last_timestamp is None:
        raise RuntimeError("Existing archive not found.")

    last_time = datetime.fromtimestamp(
        last_timestamp / 1000,
        tz=timezone.utc
    )

    print(f"\nLast Archived UTC: {last_time.isoformat()}")

    candles = fetch_new_candles(
        market="SPOT",
        timeframe="5m",
        last_timestamp=last_timestamp,
        page_size=100
    )

    print("\nIncremental Fetch Results")
    print("=========================")
    print(f"New Candles Retrieved: {len(candles)}")

    if candles:
        result = save_candles(candles, filename)

        print(f"New Candles Saved: {result['new_candles']}")
        print(f"Total Archived Candles: {result['total_candles']}")

        newest_time = datetime.fromtimestamp(
            candles[-1]["timestamp"] / 1000,
            tz=timezone.utc
        )

        print(f"Newest Candle UTC: {newest_time.isoformat()}")

    else:
        print("No new completed candles found.")
        print("Archive remains unchanged.")

    print("\nINCREMENTAL UPDATE COMPLETED")
    print("LIVE TRADING: DISABLED")


if __name__ == "__main__":
    main()
