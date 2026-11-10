
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

ARCHIVE_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Archive"
    / "BTC_USDT_SPOT_5m.csv"
)

INTERVAL_MS = 5 * 60 * 1000
TEST_CANDLE_COUNT = 5
API_TIMEOUT = 30


def load_archive():
    print("Loading historical archive...", flush=True)

    if not ARCHIVE_FILE.exists():
        raise RuntimeError(
            f"Archive file not found: {ARCHIVE_FILE}"
        )

    records = {}

    with ARCHIVE_FILE.open(
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
            raise RuntimeError("Archive CSV has no header.")

        missing_fields = required_fields - set(reader.fieldnames)

        if missing_fields:
            raise RuntimeError(
                f"Missing CSV columns: {sorted(missing_fields)}"
            )

        for row in reader:
            timestamp = int(row["timestamp"])

            if timestamp in records:
                raise RuntimeError(
                    f"Duplicate archive timestamp: {timestamp}"
                )

            records[timestamp] = row

    print(
        f"Archive candles loaded: {len(records)}",
        flush=True
    )

    return records


def select_test_candles(archive):
    timestamps = sorted(archive.keys())

    if len(timestamps) < TEST_CANDLE_COUNT:
        raise RuntimeError(
            "Not enough archived candles for recovery test."
        )

    middle = len(timestamps) // 2

    selected = timestamps[
        middle:middle + TEST_CANDLE_COUNT
    ]

    if len(selected) != TEST_CANDLE_COUNT:
        raise RuntimeError(
            "Could not select the required test candles."
        )

    for index in range(1, len(selected)):
        difference = (
            selected[index] - selected[index - 1]
        )

        if difference != INTERVAL_MS:
            raise RuntimeError(
                "Selected archive candles are not consecutive."
            )

    return selected


def fetch_test_range(start_timestamp, end_timestamp):
    print("Requesting historical candles from OKX...", flush=True)

    params = {
        "instId": MARKET_INSTRUMENTS["SPOT"],
        "bar": TIMEFRAME_MAPPING["5m"],
        "after": str(end_timestamp + 1),
        "before": str(start_timestamp - 1),
        "limit": "100"
    }

    response = requests.get(
        f"{BASE_URL}/api/v5/market/history-candles",
        params=params,
        timeout=API_TIMEOUT
    )

    print(
        f"OKX HTTP Status: {response.status_code}",
        flush=True
    )

    response.raise_for_status()

    result = response.json()

    if result.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {result.get('msg')}"
        )

    raw_data = result.get("data", [])

    print(
        f"Raw API Candles: {len(raw_data)}",
        flush=True
    )

    candles = parse_candles(
        raw_data,
        "SPOT",
        "5m"
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


def compare_candles(archive, recovered, selected):
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
            archived_value = float(archived_row[field])
            api_value = float(api_row[field])

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


def main():
    print("\nHISTORICAL RECOVERY TEST", flush=True)
    print("========================", flush=True)
    print("Provider: OKX", flush=True)
    print("Market: SPOT", flush=True)
    print("Symbol: BTC-USDT", flush=True)
    print("Timeframe: 5m", flush=True)
    print("Archive Modification: DISABLED", flush=True)

    archive = load_archive()

    selected = select_test_candles(archive)

    start_timestamp = selected[0]
    end_timestamp = selected[-1]

    print(
        f"\nExpected Candles: {TEST_CANDLE_COUNT}",
        flush=True
    )
    print(
        f"Test Start UTC: {start_timestamp}",
        flush=True
    )
    print(
        f"Test End UTC: {end_timestamp}",
        flush=True
    )

    recovered = fetch_test_range(
        start_timestamp,
        end_timestamp
    )

    missing, mismatches = compare_candles(
        archive,
        recovered,
        selected
    )

    passed = (
        len(recovered) == TEST_CANDLE_COUNT
        and not missing
        and not mismatches
    )

    print("\nRECOVERY TEST RESULTS", flush=True)
    print("=====================", flush=True)
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
        f"RECOVERY TEST STATUS: {'PASS' if passed else 'FAIL'}",
        flush=True
    )

    print("ARCHIVE MODIFIED: NO", flush=True)

    if not passed:
        raise RuntimeError(
            "Historical recovery validation failed."
        )


if __name__ == "__main__":
    main()
