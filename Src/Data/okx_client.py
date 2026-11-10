
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
        raise ValueError("Limit must be between 1 and 300.")

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
        raise RuntimeError(
            f"OKX API error: {result.get('msg')}"
        )

    candles = parse_candles(
        result.get("data", []),
        market,
        timeframe
    )

    candles.sort(key=lambda candle: candle["timestamp"])

    return candles


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
        raise ValueError("Total candles must be positive.")

    if not 1 <= page_size <= 300:
        raise ValueError("Page size must be between 1 and 300.")

    collected = {}
    after_timestamp = None
    previous_oldest = None
    page_count = 0

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
            raise RuntimeError(
                f"OKX history API error: {result.get('msg')}"
            )

        raw_data = result.get("data", [])

        if not raw_data:
            break

        page = parse_candles(raw_data, market, timeframe)

        if not page:
            break

        page.sort(key=lambda candle: candle["timestamp"])

        oldest_timestamp = page[0]["timestamp"]

        if previous_oldest is not None:
            if oldest_timestamp >= previous_oldest:
                raise RuntimeError(
                    "Pagination did not move to older candles."
                )

        for candle in page:
            collected[candle["timestamp"]] = candle

        previous_oldest = oldest_timestamp
        after_timestamp = oldest_timestamp
        page_count += 1

        if len(raw_data) < page_size:
            break

        if page_count > 100:
            raise RuntimeError("Pagination safety limit reached.")

    candles = sorted(
        collected.values(),
        key=lambda candle: candle["timestamp"]
    )

    return candles[-total_candles:]
