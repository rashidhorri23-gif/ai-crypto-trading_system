
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings


BASE_URL = "https://www.okx.com"


def validate_settings():
    print("AI Crypto Trading System")
    print("========================")
    print(f"Version: {settings.PROJECT_VERSION}")
    print(f"Execution Mode: {settings.EXECUTION_MODE}")
    print(f"Data Provider: {settings.DATA_PROVIDER}")

    if settings.TRADING_ENABLED:
        raise RuntimeError("Trading must remain disabled.")

    if settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError("Live orders must remain disabled.")

    print("Live trading: DISABLED")


def test_okx_endpoint(name, path, params):
    url = BASE_URL + path

    print(f"\nTesting: {name}")

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()
    result = response.json()

    if result.get("code") != "0":
        raise RuntimeError(
            f"OKX API error: {result.get('code')} - {result.get('msg')}"
        )

    data = result.get("data", [])

    if not data:
        raise RuntimeError("OKX returned empty data.")

    print(f"HTTP Status: {response.status_code}")
    print(f"API Code: {result.get('code')}")

    if "ticker" in name.lower():
        print(f"Instrument: {data[0].get('instId')}")
        print(f"Last Price: {data[0].get('last')}")

    if "candles" in name.lower():
        print(f"Returned Candles: {len(data)}")
        print(f"Latest Candle Timestamp: {data[0][0]}")
        print(f"Latest Candle Close: {data[0][4]}")

    print(f"{name}: SUCCESS")


def test_okx_public_data():
    tests = [
        (
            "SPOT Ticker",
            "/api/v5/market/ticker",
            {"instId": "BTC-USDT"}
        ),
        (
            "SPOT Candles",
            "/api/v5/market/candles",
            {"instId": "BTC-USDT", "bar": "5m", "limit": "3"}
        ),
        (
            "SWAP Ticker",
            "/api/v5/market/ticker",
            {"instId": "BTC-USDT-SWAP"}
        ),
        (
            "SWAP Candles",
            "/api/v5/market/candles",
            {"instId": "BTC-USDT-SWAP", "bar": "5m", "limit": "3"}
        )
    ]

    failures = []

    for name, path, params in tests:
        try:
            test_okx_endpoint(name, path, params)
        except Exception as error:
            print(f"{name}: FAILED")
            print(f"Error: {error}")
            failures.append(name)

    if failures:
        raise RuntimeError(
            "OKX connection test failed: " + ", ".join(failures)
        )

    print("\n========================")
    print("ALL OKX TESTS PASSED")
    print("SPOT: CONNECTED")
    print("SWAP: CONNECTED")
    print("Trading: DISABLED")


def main():
    validate_settings()
    test_okx_public_data()


if __name__ == "__main__":
    main()
