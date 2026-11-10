
def fetch_test_range(start_timestamp, end_timestamp):
    params = {
        "instId": MARKET_INSTRUMENTS["SPOT"],
        "bar": TIMEFRAME_MAPPING["5m"],
        "after": str(end_timestamp + 1),
        "before": str(start_timestamp - 1),
        "limit": "20"
    }

    response = requests.get(
        f"{BASE_URL}/api/v5/market/history-candles",
        params=params,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if result.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {result.get('msg')}"
        )

    candles = parse_candles(
        result.get("data", []),
        "SPOT",
        "5m"
    )

    return sorted(
        [
            candle
            for candle in candles
            if start_timestamp
            <= candle["timestamp"]
            <= end_timestamp
        ],
        key=lambda candle: candle["timestamp"]
    )
