
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from Src.Data.okx_client import fetch_new_candles


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT_DIR / "Data" / "Archive" / "BTC_USDT_SPOT_4H.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

CANDLE_INTERVAL_MS = 4 * 60 * 60 * 1000


def validate_archive(df):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Archive is missing required columns: {missing}"
        )

    df = df.copy()

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="raise"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Archive contains invalid timestamps.")

    if (df["timestamp"] % 1 != 0).any():
        raise ValueError("Archive contains non-integer timestamps.")

    df["timestamp"] = df["timestamp"].astype("int64")

    if df["timestamp"].duplicated().any():
        raise ValueError("Archive contains duplicate timestamps.")

    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def update_btc_4h_archive():
    print("=" * 60)
    print("BTC 4H OKX DATA UPDATE")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"BTC 4H archive not found: {DATA_FILE}"
        )

    existing = pd.read_csv(DATA_FILE)
    existing = validate_archive(existing)

    last_timestamp = int(existing["timestamp"].iloc[-1])

    last_time = datetime.fromtimestamp(
        last_timestamp / 1000,
        tz=timezone.utc
    )

    print(f"Existing candles: {len(existing)}")
    print(f"Last archived candle: {last_time.isoformat()}")

    new_candles = fetch_new_candles(
        market="SPOT",
        timeframe="4H",
        last_timestamp=last_timestamp,
        page_size=100
    )

    if not new_candles:
        print("No new confirmed candles returned by OKX.")
        return 0

    new_rows = []

    for candle in new_candles:
        row = {
            column: candle.get(column, pd.NA)
            for column in existing.columns
        }
        new_rows.append(row)

    new_df = pd.DataFrame(
        new_rows,
        columns=existing.columns
    )

    new_df = validate_archive(new_df)

    if (new_df["timestamp"] <= last_timestamp).any():
        raise ValueError(
            "OKX returned a candle that is not newer than the archive."
        )

    combined = pd.concat(
        [existing, new_df],
        ignore_index=True
    )

    combined = validate_archive(combined)

    timestamp_diffs = combined["timestamp"].diff().dropna()

    if not timestamp_diffs.empty:
        invalid_intervals = timestamp_diffs[
            timestamp_diffs != CANDLE_INTERVAL_MS
        ]

        if not invalid_intervals.empty:
            raise ValueError(
                "Archive contains missing or irregular 4H intervals. "
                "The updated file was not saved."
            )

    temp_file = DATA_FILE.with_name(
        DATA_FILE.name + ".tmp"
    )

    combined.to_csv(
        temp_file,
        index=False
    )

    temp_file.replace(DATA_FILE)

    latest_timestamp = int(combined["timestamp"].iloc[-1])

    latest_time = datetime.fromtimestamp(
        latest_timestamp / 1000,
        tz=timezone.utc
    )

    print(f"New confirmed candles: {len(new_df)}")
    print(f"Total candles: {len(combined)}")
    print(f"Latest archived candle: {latest_time.isoformat()}")
    print(f"Updated archive: {DATA_FILE}")
    print("=" * 60)

    return len(new_df)


if __name__ == "__main__":
    try:
        update_btc_4h_archive()
    except Exception as error:
        print(f"DATA UPDATE FAILED: {error}")
        sys.exit(1)
