
import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings
from Src.Data.okx_client import (
    fetch_historical_candles,
    fetch_new_candles
)
from Src.Data.archive import (
    save_candles,
    get_latest_timestamp
)

from Src.Analysis.daily_structure import analyze_daily_structure
from Src.Analysis.market_direction import determine_market_direction


MARKET = "SPOT"

TIMEFRAME_CONFIG = {
    "1D": {
        "target": 365,
        "interval_ms": 86400000,
        "filename": "BTC_USDT_SPOT_1D.csv"
    },
    "4H": {
        "target": 540,
        "interval_ms": 14400000,
        "filename": "BTC_USDT_SPOT_4H.csv"
    },
    "1H": {
        "target": 720,
        "interval_ms": 3600000,
        "filename": "BTC_USDT_SPOT_1H.csv"
    },
    "15m": {
        "target": 960,
        "interval_ms": 900000,
        "filename": "BTC_USDT_SPOT_15m.csv"
    },
    "5m": {
        "target": 1016,
        "interval_ms": 300000,
        "filename": "BTC_USDT_SPOT_5m.csv"
    }
}


def get_archive_path(filename):
    return PROJECT_ROOT / "Data" / "Archive" / filename


def count_archive_rows(filename):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        return 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        return sum(1 for row in reader)


def initialize_archive(timeframe, config):
    filename = config["filename"]
    target = config["target"]
    archive_path = get_archive_path(filename)

    if timeframe == "5m" and not archive_path.exists():
        raise RuntimeError(
            "Existing 5m archive is missing. "
            "Automatic recreation is disabled."
        )

    current_count = count_archive_rows(filename)

    if current_count >= target:
        print(
            f"{timeframe}: Existing archive contains "
            f"{current_count} candles. Initial seed not required."
        )
        return

    print(
        f"{timeframe}: Initializing historical archive. "
        f"Target: {target} candles."
    )

    candles = fetch_historical_candles(
        market=MARKET,
        timeframe=timeframe,
        total_candles=target,
        page_size=100
    )

    if len(candles) != target:
        raise RuntimeError(
            f"{timeframe}: Historical seed incomplete. "
            f"Expected {target}, received {len(candles)}. "
            "Archive was not updated."
        )

    result = save_candles(candles, filename)

    print(
        f"{timeframe}: Historical seed saved. "
        f"New candles: {result['new_candles']}, "
        f"total candles: {result['total_candles']}"
    )


def update_archive(timeframe, config):
    filename = config["filename"]

    last_timestamp = get_latest_timestamp(filename)

    if last_timestamp is None:
        raise RuntimeError(
            f"{timeframe}: Archive has no latest timestamp."
        )

    last_time = datetime.fromtimestamp(
        last_timestamp / 1000,
        tz=timezone.utc
    )

    print(f"\n{timeframe} INCREMENTAL UPDATE")
    print("============================")
    print(f"Last Archived UTC: {last_time.isoformat()}")

    candles = fetch_new_candles(
        market=MARKET,
        timeframe=timeframe,
        last_timestamp=last_timestamp,
        page_size=100
    )

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


def audit_archive(filename, interval_ms):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        raise RuntimeError(
            f"Archive file not found: {filename}"
        )

    timestamps = []
    invalid_rows = 0
    misaligned_timestamps = 0
    total_rows = 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            try:
                timestamp = int(row["timestamp"])

                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
                volume = float(row["volume"])

                if (
                    min(
                        open_price,
                        high_price,
                        low_price,
                        close_price
                    ) <= 0
                    or volume < 0
                    or high_price < max(
                        open_price,
                        close_price,
                        low_price
                    )
                    or low_price > min(
                        open_price,
                        close_price,
                        high_price
                    )
                ):
                    invalid_rows += 1

                timestamps.append(timestamp)

                if timestamp % interval_ms != 0:
                    misaligned_timestamps += 1

            except (ValueError, KeyError, TypeError):
                invalid_rows += 1

    if not timestamps:
        raise RuntimeError(
            f"{filename}: Archive contains no valid timestamps."
        )

    unique_timestamps = sorted(set(timestamps))

    duplicate_count = (
        len(timestamps) - len(unique_timestamps)
    )

    missing_candles = 0
    gap_count = 0

    for index in range(1, len(unique_timestamps)):
        difference = (
            unique_timestamps[index]
            - unique_timestamps[index - 1]
        )

        if difference > interval_ms:
            missing = (
                difference // interval_ms
            ) - 1

            missing_candles += missing
            gap_count += 1

        elif difference < interval_ms:
            invalid_rows += 1

    first_time = datetime.fromtimestamp(
        unique_timestamps[0] / 1000,
        tz=timezone.utc
    )

    last_time = datetime.fromtimestamp(
        unique_timestamps[-1] / 1000,
        tz=timezone.utc
    )

    expected_count = (
        (unique_timestamps[-1] - unique_timestamps[0])
        // interval_ms
    ) + 1

    continuous = (
        len(unique_timestamps) == expected_count
    )

    passed = (
        duplicate_count == 0
        and missing_candles == 0
        and invalid_rows == 0
        and misaligned_timestamps == 0
        and continuous
    )

    print("\nARCHIVE AUDIT REPORT")
    print("====================")
    print(f"Archive File: {filename}")
    print(f"Total Rows: {total_rows}")
    print(f"Unique Candles: {len(unique_timestamps)}")
    print(f"Duplicate Rows: {duplicate_count}")
    print(f"Missing Candles: {missing_candles}")
    print(f"Gap Segments: {gap_count}")
    print(f"Invalid Rows: {invalid_rows}")
    print(f"Misaligned Timestamps: {misaligned_timestamps}")
    print(f"First Candle UTC: {first_time.isoformat()}")
    print(f"Last Candle UTC: {last_time.isoformat()}")
    print(f"Expected Continuous Count: {expected_count}")
    print(f"Continuous Timeline: {continuous}")
    print(f"AUDIT STATUS: {'PASS' if passed else 'ISSUES DETECTED'}")

    return passed


# ============================================================
# MARKET DIRECTION
# ============================================================

def load_structure_candles(filename, limit):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        raise RuntimeError(
            f"Structure archive not found: {filename}"
        )

    candles = []

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                candle = {
                    "timestamp": int(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                }

                candles.append(candle)

            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"Invalid candle in {filename}: {error}"
                )

    candles.sort(key=lambda candle: candle["timestamp"])

    if len(candles) < limit:
        raise RuntimeError(
            f"{filename}: Insufficient candles. "
            f"Required {limit}, found {len(candles)}."
        )

    return candles[-limit:]


def run_market_direction_analysis():
    print("\nREAL MARKET DIRECTION ANALYSIS")
    print("==============================")

    daily_candles = load_structure_candles(
        TIMEFRAME_CONFIG["1D"]["filename"],
        90
    )

    four_hour_candles = load_structure_candles(
        TIMEFRAME_CONFIG["4H"]["filename"],
        540
    )

    print(f"Daily Candles: {len(daily_candles)}")
    print(f"4H Candles: {len(four_hour_candles)}")

    daily_result = analyze_daily_structure(daily_candles)

    four_hour_result = analyze_daily_structure(
        four_hour_candles
    )

    decision = determine_market_direction(
        daily_result,
        four_hour_result
    )

    print("\nDAILY MARKET STRUCTURE")
    print("======================")
    print("Trend:", decision["daily_bias"])
    print("Event Bias:", decision["daily_event_bias"])
    print("Reason:", decision["daily_structure_reason"])

    print("\n4H MARKET STRUCTURE")
    print("==================")
    print("Trend:", decision["four_hour_bias"])
    print("Event Bias:", decision["four_hour_event_bias"])
    print("Reason:", decision["four_hour_structure_reason"])

    print("\nFINAL MARKET DIRECTION")
    print("======================")
    print("Allowed Direction:", decision["allowed_direction"])
    print("Status:", decision["status"])
    print("Reason:", decision["reason"])
    print(
        "Trade Directions:",
        decision["trade_directions_to_evaluate"]
    )

    return decision


def main():
    print("AI CRYPTO TRADING SYSTEM")
    print("========================")
    print(f"Provider: {settings.DATA_PROVIDER}")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Timeframes: 1D, 4H, 1H, 15m, 5m")
    print("Mode: MULTI-TIMEFRAME ARCHIVE + AUDIT + DIRECTION")
    print("Live Trading: DISABLED")

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError(
            "Live trading must remain disabled."
        )

    for timeframe, config in TIMEFRAME_CONFIG.items():
        initialize_archive(timeframe, config)

    for timeframe, config in TIMEFRAME_CONFIG.items():
        update_archive(timeframe, config)

    print("\nMULTI-TIMEFRAME AUDIT")
    print("=====================")

    audit_results = {}

    for timeframe, config in TIMEFRAME_CONFIG.items():
        print(f"\nTIMEFRAME: {timeframe}")

        audit_results[timeframe] = audit_archive(
            config["filename"],
            config["interval_ms"]
        )

    all_passed = all(audit_results.values())

    print("\nFINAL SYSTEM REPORT")
    print("===================")

    for timeframe, passed in audit_results.items():
        print(
            f"{timeframe}: {'PASS' if passed else 'ISSUES DETECTED'}"
        )

    print(
        f"OVERALL AUDIT: {'PASS' if all_passed else 'ISSUES DETECTED'}"
    )

    print("LIVE TRADING: DISABLED")

    if not all_passed:
        raise RuntimeError(
            "One or more timeframe archives failed validation."
        )

    # Run market direction only after all archives pass validation.
    market_decision = run_market_direction_analysis()

    if market_decision["allowed_direction"] == "WAIT":
        print("\nSETUP GENERATION BLOCKED")
        print("========================")
        print("Market direction is not aligned.")
        print("No trade direction is currently allowed.")

    else:
        print("\nSETUP DIRECTION AUTHORIZED")
        print("==========================")
        print(
            "Authorized Direction:",
            market_decision["allowed_direction"]
        )
        print(
            "The 4H setup engine can now evaluate this direction."
        )

    print("\nLIVE TRADING: DISABLED")


if __name__ == "__main__":
    main()
