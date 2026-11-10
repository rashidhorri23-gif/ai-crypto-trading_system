
def fetch_new_candles(
    market,
    timeframe,
    last_timestamp,
    page_size=100
):
    if market not in MARKET_INSTRUMENTS:
        raise ValueError("Unsupported market.")

    if timeframe not in TIMEFRAME_MAPPING:
        raise ValueError("Unsupported timeframe.")

    if last_timestamp <= 0:
        raise ValueError("Invalid last timestamp.")

    if not 1 <= page_size <= 100:
        raise ValueError("Page size must be between 1 and 100.")

    collected = {}
    after_timestamp = None
    previous_oldest = None
    page_count = 0

    while True:
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

        oldest_timestamp = min(
            int(item[0]) for item in raw_data
        )

        if previous_oldest is not None:
            if oldest_timestamp >= previous_oldest:
                raise RuntimeError(
                    "Historical pagination did not move backward."
                )

        page = parse_candles(
            raw_data,
            market,
            timeframe
        )

        for candle in page:
            if candle["timestamp"] > last_timestamp:
                collected[candle["timestamp"]] = candle

        page_count += 1

        if oldest_timestamp <= last_timestamp:
            break

        if len(raw_data) < page_size:
            break

        previous_oldest = oldest_timestamp
        after_timestamp = oldest_timestamp

        if page_count >= 1000:
            raise RuntimeError(
                "Historical pagination safety limit reached."
            )

    return sorted(
        collected.values(),
        key=lambda candle: candle["timestamp"]
    )
