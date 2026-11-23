import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from Src.Data.okx_client import fetch_new_candles


ROOT_DIR = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = ROOT_DIR / "Data" / "Archive"

ARCHIVES = {
    "5m": {
        "file": "BTC_USDT_SPOT_5m.csv",
        "interval_ms": 5 * 60 * 1000,
        "page_size": 100,
    },
    "15m": {
        "file": "BTC_USDT_SPOT_15m.csv",
        "interval_ms": 15 * 60 * 1000,
        "page_size": 100,
    },
    "4H": {
        "file": "BTC_USDT_SPOT_4H.csv",
        "interval_ms": 4 * 60 * 60 * 1000,
        "page_size": 100,
    },
}

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_archive(df, timeframe):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{timeframe} archive is missing columns: {missing}"
        )

    df = df.copy()

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="raise"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            f"{timeframe} archive contains invalid timestamps."
        )

    if (df["timestamp"] % 1 != 0).any():
        raise ValueError(
            f"{timeframe} archive contains non-integer timestamps."
        )

    df["timestamp"] = df["timestamp"].astype("int64")

    if (df["timestamp"] <= 0).any():
        raise ValueError(
            f"{timeframe} archive contains invalid timestamps."
        )

    if df["timestamp"].duplicated().any():
        raise ValueError(
            f"{timeframe} archive contains duplicate timestamps."
        )

    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

        if not df[column].map(
            lambda value: pd.notna(value)
            and value >= 0
            and value < float("inf")
        ).all():
            raise ValueError(
                f"{timeframe} archive contains invalid {column} values."
            )

    if (
        (df["open"] <= 0).any()
        or (df["high"] <= 0).any()
        or (df["low"] <= 0).any()
        or (df["close"] <= 0).any()
    ):
        raise ValueError(
            f"{timeframe} archive contains non-positive prices."
        )

    if (
        df["high"]
        < df[["open", "close", "low"]].max(axis=1)
    ).any():
        raise ValueError(
            f"{timeframe} archive contains invalid high values."
        )

    if (
        df["low"]
        > df[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError(
            f"{timeframe} archive contains invalid low values."
        )

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def timestamp_to_utc(timestamp):
    return datetime.fromtimestamp(
        timestamp / 1000,
        tz=timezone.utc
    ).isoformat()


def update_archive(timeframe, settings):
    data_file = ARCHIVE_DIR / settings["file"]
    interval_ms = settings["interval_ms"]

    print()
    print("=" * 65)
    print(f"BTC {timeframe} OKX DATA UPDATE")
    print("=" * 65)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Archive not found: {data_file}"
        )

    existing = pd.read_csv(data_file)

    if existing.empty:
        raise ValueError(
            f"{timeframe} archive is empty."
        )

    existing = validate_archive(
        existing,
        timeframe
    )

    last_timestamp = int(
        existing["timestamp"].iloc[-1]
    )

    print(f"Existing candles: {len(existing)}")
    print(
        "Last archived candle: "
        f"{timestamp_to_utc(last_timestamp)}"
    )

    new_candles = fetch_new_candles(
        market="SPOT",
        timeframe=timeframe,
        last_timestamp=last_timestamp,
        page_size=settings["page_size"]
    )

    if not new_candles:
        print("No new confirmed candles returned by OKX.")
        return 0

    new_df = pd.DataFrame(new_candles)

    new_df = new_df[
        REQUIRED_COLUMNS
        + [
            column
            for column in (
                "market",
                "symbol",
                "timeframe",
                "source"
            )
            if column in new_df.columns
        ]
    ]

    new_df = validate_archive(
        new_df,
        timeframe
    )

    if (new_df["timestamp"] <= last_timestamp).any():
        raise ValueError(
            f"{timeframe}: OKX returned an old or duplicate candle."
        )

    expected_timestamp = last_timestamp + interval_ms

    if int(new_df["timestamp"].iloc[0]) != expected_timestamp:
        raise ValueError(
            f"{timeframe}: Data gap detected. "
            f"Expected {expected_timestamp}, "
            f"received {int(new_df['timestamp'].iloc[0])}. "
            "Archive was not modified."
        )

    differences = new_df["timestamp"].diff().dropna()

    if not differences.empty:
        if not (differences == interval_ms).all():
            raise ValueError(
                f"{timeframe}: New candles contain missing intervals. "
                "Archive was not modified."
            )

    for column in existing.columns:
        if column not in new_df.columns:
            new_df[column] = pd.NA

    new_df = new_df[existing.columns]

    combined = pd.concat(
        [existing, new_df],
        ignore_index=True
    )

    combined = validate_archive(
        combined,
        timeframe
    )

    temp_file = data_file.with_name(
        data_file.name + ".tmp"
    )

    combined.to_csv(
        temp_file,
        index=False
    )

    temp_file.replace(data_file)

    latest_timestamp = int(
        combined["timestamp"].iloc[-1]
    )

    print(f"New confirmed candles: {len(new_df)}")
    print(f"Total candles: {len(combined)}")
    print(
        "Latest archived candle: "
        f"{timestamp_to_utc(latest_timestamp)}"
    )
    print(f"Updated archive: {data_file}")

    return len(new_df)


def main():
    print("=" * 65)
    print("BTC MULTI-TIMEFRAME OKX DATA UPDATE")
    print("=" * 65)

    results = {}
    failures = []

    for timeframe, settings in ARCHIVES.items():
        try:
            results[timeframe] = update_archive(
                timeframe,
                settings
            )
        except Exception as error:
            failures.append(
                f"{timeframe}: {error}"
            )
            print(
                f"UPDATE FAILED [{timeframe}]: {error}"
            )

    print()
    print("=" * 65)
    print("UPDATE SUMMARY")
    print("=" * 65)

    for timeframe, count in results.items():
        print(
            f"{timeframe}: {count} new confirmed candles"
        )

    if failures:
        print()
        print("FAILED TIMEFRAMES:")

        for failure in failures:
            print(f"- {failure}")

        sys.exit(1)

    print("All timeframes updated successfully.")


if __name__ == "__main__":
    main()
