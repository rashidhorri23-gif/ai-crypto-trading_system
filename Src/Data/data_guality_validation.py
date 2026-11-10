
import csv
import math
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAME_CONFIG = {
    "5m": {
        "interval_ms": 300000,
        "filename": "BTC_USDT_SPOT_5m.csv",
        "price_jump_limit": 0.03
    },
    "15m": {
        "interval_ms": 900000,
        "filename": "BTC_USDT_SPOT_15m.csv",
        "price_jump_limit": 0.05
    },
    "1H": {
        "interval_ms": 3600000,
        "filename": "BTC_USDT_SPOT_1H.csv",
        "price_jump_limit": 0.08
    },
    "4H": {
        "interval_ms": 14400000,
        "filename": "BTC_USDT_SPOT_4H.csv",
        "price_jump_limit": 0.12
    },
    "1D": {
        "interval_ms": 86400000,
        "filename": "BTC_USDT_SPOT_1D.csv",
        "price_jump_limit": 0.20
    }
}

AGGREGATION_PAIRS = [
    ("5m", "15m"),
    ("15m", "1H"),
    ("1H", "4H"),
    ("4H", "1D")
]

PRICE_FIELDS = [
    "open",
    "high",
    "low",
    "close"
]

OHLCV_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]


def load_archive(timeframe, config):
    path = ARCHIVE_DIR / config["filename"]

    if not path.exists():
        raise RuntimeError(
            f"{timeframe}: Archive file not found."
        )

    records = {}
    errors = {
        "invalid_rows": 0,
        "duplicates": 0,
        "misaligned": 0,
        "incomplete": 0
    }

    with path.open(
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
                f"{timeframe}: Missing columns: "
                f"{sorted(missing_fields)}"
            )

        for row in reader:
            try:
                timestamp = int(row["timestamp"])

                values = {
                    field: float(row[field])
                    for field in OHLCV_FIELDS
                }

                if not all(
                    math.isfinite(value)
                    for value in values.values()
                ):
                    errors["invalid_rows"] += 1
                    continue

                open_price = values["open"]
                high_price = values["high"]
                low_price = values["low"]
                close_price = values["close"]
                volume = values["volume"]

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
                    errors["invalid_rows"] += 1
                    continue

                if timestamp % config["interval_ms"] != 0:
                    errors["misaligned"] += 1
                    continue

                if timestamp in records:
                    errors["duplicates"] += 1
                    continue

                records[timestamp] = values

            except (
                ValueError,
                KeyError,
                TypeError,
                OverflowError
            ):
                errors["invalid_rows"] += 1

    return records, errors


def audit_timeline(records, interval_ms):
    timestamps = sorted(records)

    missing_candles = 0
    gap_segments = 0

    for index in range(1, len(timestamps)):
        difference = (
            timestamps[index] - timestamps[index - 1]
        )

        if difference > interval_ms:
            missing_candles += (
                difference // interval_ms
            ) - 1

            gap_segments += 1

    if not timestamps:
        return {
            "first_timestamp": None,
            "last_timestamp": None,
            "missing_candles": 0,
            "gap_segments": 0,
            "continuous": False
        }

    expected_count = (
        (timestamps[-1] - timestamps[0])
        // interval_ms
    ) + 1

    continuous = len(timestamps) == expected_count

    return {
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "missing_candles": missing_candles,
        "gap_segments": gap_segments,
        "continuous": continuous
    }


def detect_outliers(records, timeframe, config):
    timestamps = sorted(records)

    price_jumps = []
    volume_spikes = []

    threshold = config["price_jump_limit"]

    for index in range(1, len(timestamps)):
        previous = records[timestamps[index - 1]]
        current = records[timestamps[index]]

        previous_close = previous["close"]

        change = abs(
            current["close"] / previous_close - 1
        )

        if change > threshold:
            price_jumps.append({
                "timestamp": timestamps[index],
                "change_percent": change * 100
            })

    for index in range(20, len(timestamps)):
        previous_volumes = [
            records[timestamps[position]]["volume"]
            for position in range(index - 20, index)
        ]

        positive_volumes = [
            value
            for value in previous_volumes
            if value > 0
        ]

        if not positive_volumes:
            continue

        sorted_volumes = sorted(positive_volumes)
        middle = len(sorted_volumes) // 2

        if len(sorted_volumes) % 2:
            median_volume = sorted_volumes[middle]
        else:
            median_volume = (
                sorted_volumes[middle - 1]
                + sorted_volumes[middle]
            ) / 2

        current_volume = records[
            timestamps[index]
        ]["volume"]

        if (
            median_volume > 0
            and current_volume > median_volume * 20
        ):
            volume_spikes.append({
                "timestamp": timestamps[index],
                "volume_ratio": (
                    current_volume / median_volume
                )
            })

    return price_jumps, volume_spikes


def validate_aggregation(
    lower_timeframe,
    higher_timeframe,
    archives
):
    lower_interval = TIMEFRAME_CONFIG[
        lower_timeframe
    ]["interval_ms"]

    higher_interval = TIMEFRAME_CONFIG[
        higher_timeframe
    ]["interval_ms"]

    factor = higher_interval // lower_interval

    lower_records = archives[lower_timeframe]
    higher_records = archives[higher_timeframe]

    checked = 0
    skipped = 0
    mismatches = []

    for parent_timestamp, parent in sorted(
        higher_records.items()
    ):
        expected_timestamps = [
            parent_timestamp + index * lower_interval
            for index in range(factor)
        ]

        if not all(
            timestamp in lower_records
            for timestamp in expected_timestamps
        ):
            skipped += 1
            continue

        children = [
            lower_records[timestamp]
            for timestamp in expected_timestamps
        ]

        aggregated = {
            "open": children[0]["open"],
            "high": max(
                candle["high"] for candle in children
            ),
            "low": min(
                candle["low"] for candle in children
            ),
            "close": children[-1]["close"],
            "volume": sum(
                candle["volume"] for candle in children
            )
        }

        checked += 1

        for field in OHLCV_FIELDS:
            if not math.isclose(
                aggregated[field],
                parent[field],
                rel_tol=1e-8,
                abs_tol=1e-8
            ):
                mismatches.append({
                    "timestamp": parent_timestamp,
                    "field": field,
                    "aggregated": aggregated[field],
                    "archive": parent[field]
                })

    return {
        "checked": checked,
        "skipped": skipped,
        "mismatches": mismatches
    }


def main():
    print("FULL DATA QUALITY VALIDATION")
    print("============================")
    print("Provider: OKX")
    print("Market: SPOT")
    print("Symbol: BTC-USDT")
    print("Archive Modification: DISABLED")
    print(
        f"Validation UTC: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )

    archives = {}
    audit_results = {}
    outlier_results = {}

    for timeframe, config in TIMEFRAME_CONFIG.items():
        print(
            f"\nLOADING {timeframe}",
            flush=True
        )

        records, errors = load_archive(
            timeframe,
            config
        )

        if not records:
            raise RuntimeError(
                f"{timeframe}: No valid records found."
            )

        timeline = audit_timeline(
            records,
            config["interval_ms"]
        )

        price_jumps, volume_spikes = detect_outliers(
            records,
            timeframe,
            config
        )

        archives[timeframe] = records

        audit_results[timeframe] = {
            "errors": errors,
            "timeline": timeline
        }

        outlier_results[timeframe] = {
            "price_jumps": price_jumps,
            "volume_spikes": volume_spikes
        }

        print(f"Valid Candles: {len(records)}")
        print(f"Invalid Rows: {errors['invalid_rows']}")
        print(f"Duplicate Rows: {errors['duplicates']}")
        print(f"Misaligned Timestamps: {errors['misaligned']}")
        print(
            f"Missing Candles: "
            f"{timeline['missing_candles']}"
        )
        print(f"Gap Segments: {timeline['gap_segments']}")
        print(f"Continuous Timeline: {timeline['continuous']}")
        print(f"Price Jump Warnings: {len(price_jumps)}")
        print(f"Volume Spike Warnings: {len(volume_spikes)}")

    print("\nCROSS-TIMEFRAME AGGREGATION")
    print("===========================")

    aggregation_results = {}

    for lower, higher in AGGREGATION_PAIRS:
        result = validate_aggregation(
            lower,
            higher,
            archives
        )

        aggregation_results[(lower, higher)] = result

        print(
            f"\n{lower} -> {higher}"
        )
        print(f"Compared Candles: {result['checked']}")
        print(f"Skipped Incomplete Groups: {result['skipped']}")
        print(
            f"OHLCV Mismatches: "
            f"{len(result['mismatches'])}"
        )

        if result["mismatches"]:
            print(
                f"Mismatch Examples: "
                f"{result['mismatches'][:3]}"
            )

    print("\nFINAL DATA QUALITY REPORT")
    print("=========================")

    all_passed = True

    for timeframe in TIMEFRAME_CONFIG:
        result = audit_results[timeframe]
        errors = result["errors"]
        timeline = result["timeline"]

        passed = (
            errors["invalid_rows"] == 0
            and errors["duplicates"] == 0
            and errors["misaligned"] == 0
            and timeline["missing_candles"] == 0
            and timeline["continuous"]
        )

        print(
            f"{timeframe}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_passed = False

    for pair, result in aggregation_results.items():
        passed = (
            result["checked"] > 0
            and len(result["mismatches"]) == 0
        )

        label = f"{pair[0]} -> {pair[1]}"

        print(
            f"{label}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_passed = False

    print(
        f"\nOVERALL DATA QUALITY: "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print("ARCHIVE MODIFIED: NO")

    if not all_passed:
        raise RuntimeError(
            "Data quality validation failed."
        )


if __name__ == "__main__":
    main()
