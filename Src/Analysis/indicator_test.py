
import sys
import math
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from indicators import calculate_all_indicators


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAMES = {
    "5m": "BTC_USDT_SPOT_5m.csv",
    "15m": "BTC_USDT_SPOT_15m.csv",
    "1H": "BTC_USDT_SPOT_1H.csv",
    "4H": "BTC_USDT_SPOT_4H.csv",
    "1D": "BTC_USDT_SPOT_1D.csv"
}

INDICATOR_COLUMNS = [
    "ema_20",
    "ema_50",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_14",
    "adx_14",
    "vwap",
    "average_volume_20",
    "volume_ratio"
]


def validate_indicator_output(result):
    if len(result) < 200:
        raise ValueError("Insufficient candles for EMA200.")

    latest = result.iloc[-1]

    for column in INDICATOR_COLUMNS:
        value = latest[column]

        if pd.isna(value) or not math.isfinite(float(value)):
            raise ValueError(
                f"Invalid latest indicator: {column}"
            )

    if not 0 <= latest["rsi_14"] <= 100:
        raise ValueError("RSI is outside the valid range.")

    if not 0 <= latest["adx_14"] <= 100:
        raise ValueError("ADX is outside the valid range.")

    if latest["atr_14"] <= 0:
        raise ValueError("ATR must be positive.")

    if latest["vwap"] <= 0:
        raise ValueError("VWAP must be positive.")

    return latest


def main():
    print("TECHNICAL INDICATORS VALIDATION")
    print("================================")
    print("Provider: OKX")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Archive Modification: DISABLED")

    all_passed = True

    for timeframe, filename in TIMEFRAMES.items():
        print(f"\nTIMEFRAME: {timeframe}")
        print("--------------------------------")

        archive_path = ARCHIVE_DIR / filename

        try:
            if not archive_path.exists():
                raise FileNotFoundError(
                    f"Archive not found: {filename}"
                )

            data = pd.read_csv(archive_path)

            result = calculate_all_indicators(data)

            latest = validate_indicator_output(result)

            latest_time = datetime.fromtimestamp(
                latest["timestamp"] / 1000,
                tz=timezone.utc
            ).isoformat()

            print(f"Input Candles: {len(data)}")
            print(f"Latest Candle UTC: {latest_time}")

            for column in INDICATOR_COLUMNS:
                print(
                    f"{column}: "
                    f"{float(latest[column]):.8f}"
                )

            print(f"{timeframe} INDICATOR STATUS: PASS")

        except Exception as error:
            all_passed = False

            print(f"{timeframe} INDICATOR STATUS: FAIL")
            print(f"ERROR: {error}")

    print("\nFINAL INDICATOR REPORT")
    print("======================")

    print(
        "OVERALL INDICATOR STATUS: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print("ARCHIVE MODIFIED: NO")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
