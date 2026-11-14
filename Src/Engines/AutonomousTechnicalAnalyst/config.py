import os


class Config:
    """Central configuration for the autonomous analysis engine."""

    # AI provider credentials
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

    # AI models
    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )

    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free"
    )

    # Initial market configuration
    SYMBOL = "BTC_USDT"
    MARKET_TYPE = "SPOT"
    TIMEFRAME = "4H"

    # Project paths
    DATA_DIRECTORY = "Data/Archive"

    # Request settings
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 2

    @classmethod
    def validate_providers(cls):
        """Return which AI providers have credentials configured."""

        return {
            "gemini": bool(cls.GEMINI_API_KEY),
            "openrouter": bool(cls.OPENROUTER_API_KEY),
        }
