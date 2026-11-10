import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings


def validate_settings():
    print("AI Crypto Trading System")
    print("========================")

    print(f"Version: {settings.PROJECT_VERSION}")
    print(f"Execution Mode: {settings.EXECUTION_MODE}")
    print(f"Markets: {settings.MARKETS}")
    print(f"Assets: {settings.ASSETS}")
    print(f"Timeframes: {settings.TIMEFRAMES}")
    print(f"Timezone: {settings.TIMEZONE}")
    print(f"Data Provider: {settings.DATA_PROVIDER}")

    if settings.TRADING_ENABLED:
        raise RuntimeError("Trading must remain disabled.")

    if settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError("Live orders must remain disabled.")

    print("========================")
    print("Configuration validation successful.")
    print("Live trading: DISABLED")


def main():
    validate_settings()


if __name__ == "__main__":
    main()
