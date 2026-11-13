
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = PROJECT_ROOT / "Data" / "Archive"

TIMEFRAME_CONFIG = {
    "1H": {
        "filename": "BTC_USDT_SPOT_1H.csv",
        "interval_ms": 3600000
    },
    "4H": {
        "filename": "BTC_USDT_SPOT_4H.csv",
        "interval_ms": 14400000
    },
    "1D": {
        "filename": "BTC_USDT_SPOT_1D.csv",
        "interval_ms": 86400000
    }
}


def load_ohlcv(timeframe):
    if timeframe not in TIMEFRAME_CONFIG:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    config = TIMEFRAME_CONFIG[timeframe]
    file_path = ARCHIVE_DIR / config["filename"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data[required].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data.isna().any().any():
        raise ValueError("Archive contains missing or invalid values.")

    values = data[required].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Archive contains non-finite values.")

    data["timestamp"] = data["timestamp"].astype("int64")

    if data["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    data = data.sort_values("timestamp").reset_index(drop=True)

    interval_ms = config["interval_ms"]

    if (data["timestamp"] % interval_ms != 0).any():
        raise ValueError("Misaligned candle timestamps found.")

    price_columns = ["open", "high", "low", "close"]

    if (data[price_columns] <= 0).any().any():
        raise ValueError("Non-positive candle prices found.")

    if (data["volume"] < 0).any():
        raise ValueError("Negative volume found.")

    if (data["high"] < data[price_columns].max(axis=1)).any():
        raise ValueError("Invalid candle high values.")

    if (data["low"] > data[price_columns].min(axis=1)).any():
        raise ValueError("Invalid candle low values.")

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    data = data[
        data["timestamp"] + interval_ms <= now_ms
    ].copy()

    data = data.reset_index(drop=True)

    if data.empty:
        raise ValueError("No completed candles available.")

    return data


def load_all_timeframes():
    return {
        timeframe: load_ohlcv(timeframe)
        for timeframe in ("1D", "4H", "1H")
    }
