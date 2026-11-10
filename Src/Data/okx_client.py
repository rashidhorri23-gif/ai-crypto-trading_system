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

    candles = []

    for item in result.get("data", []):
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

    candles.sort(key=lambda candle: candle["timestamp"])

    return candles
