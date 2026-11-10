
import csv
import math
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Data.okx_client import (
    BASE_URL,
    MARKET_INSTRUMENTS,
    TIMEFRAME_MAPPING,
    parse_candles
)


MARKET = "SPOT"
TEST_CANDLE_COUNT = 5
API_TIMEOUT = 30

TIMEFRAME_CONFIG = {
    "1D": {
        "interval_ms": 86400000,
        "filename": "BTC_USDT_SPOT_1D.csv"
    },
    "4H": {
        "interval_ms": 14400000,
        "filename": "BTC_USDT_SPOT_4H.csv"
    },
    "1H": {
        "interval_ms": 3600000,
        "filename": "BTC_USDT_SPOT_1H.csv"
    },
    "15m": {
        "interval_ms": 900000,
        "filename": "BTC_USDT_SPOT_15m.csv"
    },
    "5m": {
        "interval_ms": 300000,
        "filename": "BTC_USDT_SPOT_5m.csv"
    }
}


def load_archive(timeframe, filename):
    archive_path = (
        PROJECT_ROOT
        / "Data"
        / "Archive"
        / filename
    )

    if not archive_path.exists():
        raise RuntimeError(
            f"{timeframe}: Archive file not found."
        )

    records = {}

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required_fields = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        if not reader.fieldnames:
            raise RuntimeError(
                f"{timeframe}: CSV header is missing."
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise RuntimeError(
                f"{timeframe}: Missing fields: "
                f"{sorted(missing_fields)}"
            )

        for row in reader:
            timestamp = int(row["timestamp"])

            if timestamp in records:
                raise RuntimeError(
                    f"{timeframe}: Duplicate timestamp."
                )

            records[timestamp] = row

    return records


def select_test_candles(archive, interval_ms, timeframe):
    timestamps = sorted(archive.keys())

    if len(timestamps) < TEST_CANDLE_COUNT:
        raise RuntimeError(
            f"{timeframe}: Not enough archived candles."
        )

    middle = len(timestamps) // 2

    selected = timestamps[
        middle:middle + TEST_CANDLE_COUNT
    ]

    if len(selected) != TEST_CANDLE_COUNT:
        raise RuntimeError(
            f"{timeframe}: Test selection failed."
        )

    for index in range(1, len(selected)):
        difference = (
            selected[index] - selected[index - 1]
        )

        if difference != interval_ms:
            raise RuntimeError(
                f"{timeframe}: Selected candles "
                "are not consecutive."
            )

    return selected


def fetch_test_range(timeframe, start_timestamp, end_timestamp):
    params = {
        "instId": MARKET_INSTRUMENTS[MARKET],
        "bar": TIMEFRAME_MAPPING[timeframe],
        "after": str(end_timestamp + 1),
        "before": str(start_timestamp - 1),
        "limit": "100"
    }

    response = requests.get(
        f"{BASE_URL}/api/v5/market/history-candles",
        params=params,
        timeout=API_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    if result.get("code") != "0":
        raise RuntimeError(
            f"{timeframe}: OKX API error: "
            f"{result.get('msg')}"
        )

    candles = parse_candles(
        result.get("data", []),
        MARKET,
        timeframe
    )

    filtered = [
        candle
        for candle in candles
        if start_timestamp
        <= candle["timestamp"]
        <= end_timestamp
    ]

    unique = {
        candle["timestamp"]: candle
        for candle in filtered
    }

    return sorted(
        unique.values(),
        key=lambda candle: candle["timestamp"]
    )


def compare_candles(archive, recovered, selected, timeframe):
    recovered_map = {
        candle["timestamp"]: candle
        for candle in recovered
    }

    missing = [
        timestamp
        for timestamp in selected
        if timestamp not in recovered_map
    ]

    mismatches = []

    fields = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for timestamp in selected:
        if timestamp not in recovered_map:
            continue

        archived_row = archive[timestamp]
        api_row = recovered_map[timestamp]

        for field in fields:
            archived_value = float(
                archived_row[field]
            )

            api_value = float(
                api_row[field]
            )

            if not math.isclose(
                archived_value,
                api_value,
                rel_tol=1e-12,
                abs_tol=1e-12
            ):
                mismatches.append({
                    "timestamp": timestamp,
                    "field": field,
                    "archive": archived_value,
                    "api": api_value
                })

    return missing, mismatches


def test_timeframe(timeframe, config):
    print(f"\nTIMEFRAME: {timeframe}", flush=True)
    print("-" * 35, flush=True)

    archive = load_archive(
        timeframe,
        config["filename"]
    )

    selected = select_test_candles(
        archive,
        config["interval_ms"],
        timeframe
    )

    recovered = fetch_test_range(
        timeframe,
        selected[0],
        selected[-1]
    )

    missing, mismatches = compare_candles(
        archive,
        recovered,
        selected,
        timeframe
    )

    passed = (
        len(recovered) == TEST_CANDLE_COUNT
        and not missing
        and not mismatches
    )

    print(
        f"Archive Candles: {len(archive)}",
        flush=True
    )

    print(
        f"Expected Candles: {TEST_CANDLE_COUNT}",
        flush=True
    )

    print(
        f"API Candles Retrieved: {len(recovered)}",
        flush=True
    )

    print(
        f"Missing From API: {len(missing)}",
        flush=True
    )

    print(
        f"OHLCV Mismatches: {len(mismatches)}",
        flush=True
    )

    if missing:
        print(
            f"Missing Timestamps: {missing}",
            flush=True
        )

    if mismatches:
        print(
            f"Mismatch Details: {mismatches[:5]}",
            flush=True
        )

    print(
        f"{timeframe} RECOVERY STATUS: "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True
    )

    return passed


def main():
    print("MULTI-TIMEFRAME HISTORICAL RECOVERY TEST")
    print("========================================")
    print("Provider: OKX")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Test Candles Per Timeframe: 5")
    print("Archive Modification: DISABLED")

    results = {}

    for timeframe, config in TIMEFRAME_CONFIG.items():
        results[timeframe] = test_timeframe(
            timeframe,
            config
        )

    print("\nFINAL RECOVERY REPORT")
    print("=====================")

    for timeframe, passed in results.items():
        print(
            f"{timeframe}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    all_passed = all(results.values())

    print(
        f"OVERALL RECOVERY STATUS: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print("ARCHIVE MODIFIED: NO")

    if not all_passed:
        raise RuntimeError(
            "One or more timeframe recovery tests failed."
        )


if __name__ == "__main__":
    main()
