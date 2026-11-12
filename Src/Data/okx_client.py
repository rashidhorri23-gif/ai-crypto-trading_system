
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

REQUEST_TIMEOUT = 20
MAX_PAGINATION_PAGES = 1000


def validate_market_timeframe(market, timeframe):
    if market not in MARKET_INSTRUMENTS:
        raise ValueError("Unsupported market.")

    if timeframe not in TIMEFRAME_MAPPING:
        raise ValueError("Unsupported timeframe.")


def request_candles(endpoint, params):
    response = requests.get(
        f"{BASE_URL}{endpoint}",
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise RuntimeError("Invalid OKX API response.")

    if result.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {result.get('msg', 'Unknown error')}"
        )

    data = result.get("data")

    if not isinstance(data, list):
        raise RuntimeError("Invalid candle data returned by OKX.")

    return data


def get_raw_timestamp(item):
    if not isinstance(item, (list, tuple)) or len(item) < 9:
        raise ValueError("Invalid OKX candle record.")

    try:
        timestamp = int(item[0])
    except (TypeError, ValueError):
        raise ValueError("Invalid candle timestamp returned by OKX.")

    if timestamp < 0:
        raise ValueError("Negative candle timestamp returned by OKX.")

    return timestamp


def parse_candles(data, market, timeframe):
    validate_market_timeframe(market, timeframe)

    candles = []

    for item in data:
        timestamp = get_raw_timestamp(item)

        confirmation = str(item[8]).strip()

        if confirmation != "1":
            continue

        try:
            candle = {
                "timestamp": timestamp,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "market": market,
                "symbol": MARKET_INSTRUMENTS[market],
                "timeframe": timeframe,
                "source": "OKX"
            }
        except (TypeError, ValueError, IndexError):
            raise ValueError("Invalid OHLCV values returned by OKX.")

        values = (
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"],
            candle["volume"]
        )

        if not all(
            value >= 0 and value < float("inf")
            for value in values
        ):
            raise ValueError("Non-finite or negative candle values.")

        if min(
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"]
        ) <= 0:
            raise ValueError("Candle prices must be positive.")

        if candle["high"] < max(
            candle["open"],
            candle["close"],
            candle["low"]
        ):
            raise ValueError("Invalid candle high value.")

        if candle["low"] > min(
            candle["open"],
            candle["close"],
            candle["high"]
        ):
            raise ValueError("Invalid candle low value.")

        candles.append(candle)

    return candles


def sort_and_deduplicate(candles):
    unique_candles = {}

    for candle in candles:
        unique_candles[candle["timestamp"]] = candle

    return sorted(
        unique_candles.values(),
        key=lambda candle: candle["timestamp"]
    )


def fetch_candles(market, timeframe, limit=100):
    validate_market_timeframe(market, timeframe)

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("Invalid candle limit.")

    if not 1 <= limit <= 300:
        raise ValueError("Invalid candle limit.")

    params = {
        "instId": MARKET_INSTRUMENTS[market],
        "bar": TIMEFRAME_MAPPING[timeframe],
        "limit": str(limit)
    }

    raw_data = request_candles(
        "/api/v5/market/candles",
        params
    )

    candles = parse_candles(
        raw_data,
        market,
        timeframe
    )

    return sort_and_deduplicate(candles)


def fetch_historical_candles(
    market,
    timeframe,
    total_candles=1000,
    page_size=100
):
    validate_market_timeframe(market, timeframe)

    if (
        isinstance(total_candles, bool)
        or not isinstance(total_candles, int)
        or total_candles < 1
    ):
        raise ValueError("Invalid total candle count.")

    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= 300
    ):
        raise ValueError("Invalid page size.")

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

        raw_data = request_candles(
            "/api/v5/market/history-candles",
            params
        )

        if not raw_data:
            break

        raw_timestamps = [
            get_raw_timestamp(item)
            for item in raw_data
        ]

        oldest_timestamp = min(raw_timestamps)

        if (
            previous_oldest is not None
            and oldest_timestamp >= previous_oldest
        ):
            raise RuntimeError(
                "Historical pagination did not move backward."
            )

        page = parse_candles(
            raw_data,
            market,
            timeframe
        )

        for candle in page:
            collected[candle["timestamp"]] = candle

        page_count += 1

        if len(raw_data) < page_size:
            break

        if len(collected) >= total_candles:
            break

        previous_oldest = oldest_timestamp
        after_timestamp = oldest_timestamp

        if page_count >= MAX_PAGINATION_PAGES:
            raise RuntimeError(
                "Historical pagination safety limit reached."
            )

    result = sort_and_deduplicate(
        list(collected.values())
    )

    return result[-total_candles:]


def fetch_new_candles(
    market,
    timeframe,
    last_timestamp,
    page_size=100
):
    validate_market_timeframe(market, timeframe)

    if (
        isinstance(last_timestamp, bool)
        or not isinstance(last_timestamp, int)
        or last_timestamp <= 0
    ):
        raise ValueError("Invalid last timestamp.")

    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= 100
    ):
        raise ValueError("Invalid page size.")

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

        raw_data = request_candles(
            "/api/v5/market/history-candles",
            params
        )

        if not raw_data:
            break

        raw_timestamps = [
            get_raw_timestamp(item)
            for item in raw_data
        ]

        oldest_timestamp = min(raw_timestamps)

        if (
            previous_oldest is not None
            and oldest_timestamp >= previous_oldest
        ):
            raise RuntimeError(
                "New candle pagination did not move backward."
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

        if page_count >= MAX_PAGINATION_PAGES:
            raise RuntimeError(
                "New candle pagination safety limit reached."
            )

    return sort_and_deduplicate(
        list(collected.values())
    )
