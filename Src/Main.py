
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings
from Src.Data.okx_client import fetch_historical_candles
from Src.Data.archive import save_candles


def main():
    print("AI Crypto Trading System")
    print("========================")
    print(f"Provider: {settings.DATA_PROVIDER}")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Timeframe: 5m")
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

    result = save_candles(
        candles,
        "BTC_USDT_SPOT_5m.csv"
    )

    print("\nArchive Results")
    print("===============")
    print(f"Archive File: {result['file']}")
    print(f"Previous Candles: {result['previous_candles']}")
    print(f"New Candles: {result['new_candles']}")
    print(f"Total Archived Candles: {result['total_candles']}")

    timestamps = [
        candle["timestamp"]
        for candle in candles
    ]

    oldest = datetime.fromtimestamp(
        min(timestamps) / 1000,
        tz=timezone.utc
    )

    newest = datetime.fromtimestamp(
        max(timestamps) / 1000,
        tz=timezone.utc
    )

    print(f"Fetched From UTC: {oldest.isoformat()}")
    print(f"Fetched Until UTC: {newest.isoformat()}")

    print("\nARCHIVE UPDATE COMPLETED")
    print("LIVE TRADING: DISABLED")


if __name__ == "__main__":
    main()
