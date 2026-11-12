
import requests


BASE_URL = "https://www.okx.com"

MARKET_INSTRUMENTS = {
    "SPOT": "BTC-USDT",
    "SWAP": "BTC-USDT-SWAP"
}

TIMEFRAME_MAPPING = {
    "5m": "5m",
    "15m": "15m",
    "1H": "1H",
    "4H": "4H",
    "1D": "1Dutc"
}


def parse_candles(data, market, timeframe):
    candles = []

    for item in data:
        if item[8] != "1":
            continue

        candles.append({
            "timestamp": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
            "market": market,
            "symbol": MARKET_INSTRUMENTS[market],
            "timeframe": timeframe,
            "source": "OKX"
        })

    return candles


def fetch_candles(market, timeframe, limit=100):
    if market not in MARKET_INSTRUMENTS:
        raise ValueError("Unsupported market.")

    if timeframe not in TIMEFRAME_MAPPING:
        raise ValueError("Unsupported timeframe.")

    if not 1 <= limit <= 300:
        raise ValueError("Invalid candle limit.")

    params = {
        "instId": MARKET_INSTRUMENTS[market],
        "bar": TIMEFRAME_MAPPING[timeframe],
        "limit": str(limit)
    }

    response = requests.get(
        f"{BASE_URL}/api/v5/market/candles",
        params=params,
        timeout=20
    )

    response.raise_for_status()
    result = response.json()

    if result.get("code") != "0":
        raise RuntimeError(result.get("msg"))

    candles = parse_candles(
        result.get("data", []),
        market,
        timeframe
    )

    return sorted(
        candles,
        key=lambda candle: candle["timestamp"]
    )


def fetch_historical_candles(
    market,
    timeframe,
    total_candles=1000,
    page_size=100
):
    if market not in MARKET_INSTRUMENTS:
        raise ValueError("Unsupported market.")

    if timeframe not in TIMEFRAME_MAPPING:
        raise ValueError("Unsupported timeframe.")

    if total_candles < 1:
        raise ValueError("Invalid total candle count.")

    if not 1 <= page_size <= 300:
        raise ValueError("Invalid page size.")

    collected = {}
    after_timestamp = None
    previous_oldest = None

    while len(collected) < total_candles:
        params = {
            "instId": MARKET_INSTRUMENTS[market],
            "bar": TIMEFRAME_MAPPING[timeframe],
            "limit": str(page_size)
        }

        if after_timestamp is not None:
            params["after"] = str(after_timestamp)

        response = requests.get(
            f"{BASE_URL}/api/v5/market/history-candles",
            params=params,
            timeout=20
        )

        response.raise_for_status()
        result = response.json()

        if result.get("code") != "0":
            raise RuntimeError(result.get("msg"))

        raw_data = result.get("data", [])

        if not raw_data:
            break

        page = parse_candles(
            raw_data,
            market,
            timeframe
        )

        if not page:
            break

        page.sort(key=lambda candle: candle["timestamp"])
        oldest_timestamp = page[0]["timestamp"]

        if previous_oldest is not None:
            if oldest_timestamp >= previous_oldest:
                raise RuntimeError("Pagination did not move backward.")

        for candle in page:
            collected[candle["timestamp"]] = candle

        previous_oldest = oldest_timestamp
        after_timestamp = oldest_timestamp

        if len(raw_data) < page_size:
            break

        if len(collected) >= total_candles:
            break

    return sorted(
        collected.values(),
        key=lambda candle: candle["timestamp"]
    )[-total_candles:]


def fetch_new_candles(
    market,
    timeframe,
    last_timestamp,
    page_size=300
):
    if market not in MARKET_INSTRUMENTS:
        raise ValueError("Unsupported market.")

    if timeframe not in TIMEFRAME_MAPPING:
        raise ValueError("Unsupported timeframe.")

    if last_timestamp <= 0:
        raise ValueError("Invalid last timestamp.")

    if not 1 <= page_size <= 300:
        raise ValueError("Invalid page size.")

    params = {
        "instId": MARKET_INSTRUMENTS[market],
        "bar": TIMEFRAME_MAPPING[timeframe],
        "limit": str(page_size)
    }

    response = requests.get(
        f"{BASE_URL}/api/v5/market/candles",
        params=params,
        timeout=20
    )

    response.raise_for_status()
    result = response.json()

    if result.get("code") != "0":
        raise RuntimeError(result.get("msg"))

    raw_data = result.get("data", [])

    if not raw_data:
        return []

    raw_timestamps = [
        int(item[0]) for item in raw_data
    ]

    oldest_timestamp = min(raw_timestamps)

    if (
        len(raw_data) >= page_size
        and oldest_timestamp > last_timestamp
    ):
        raise RuntimeError(
            "Incremental update exceeds the recent candle window. "
            "Historical recovery is required."
        )

    candles = parse_candles(
        raw_data,
        market,
        timeframe
    )

    new_candles = {
        candle["timestamp"]: candle
        for candle in candles
        if candle["timestamp"] > last_timestamp
    }

    return sorted(
        new_candles.values(),
        key=lambda candle: candle["timestamp"]
    )
