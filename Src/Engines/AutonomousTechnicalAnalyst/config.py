
import os


class Config:
    # API configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Primary AI model
    OPENROUTER_MODEL = "baidu/ernie-5.5-flash-preview:free"

    # Fallback AI models
    FALLBACK_MODELS = [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    ]

    # AI request settings
    MAX_OUTPUT_TOKENS = 6000
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 2
    RETRY_BASE_SECONDS = 2
    RATE_LIMIT_MAX_WAIT_SECONDS = 30
    MAX_VISION_MODELS = 6
    MIN_VISION_CONTEXT_LENGTH = 16000

    # Market configuration
    SYMBOL = "BTC_USDT"
    MARKET_TYPE = "SPOT"
    TIMEFRAME = "4H"

    # Data configuration
    DATA_DIRECTORY = "Data/Archive"

    @classmethod
    def validate(cls):
        errors = []

        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY is missing.")

        if not cls.OPENROUTER_MODEL:
            errors.append("OPENROUTER_MODEL is missing.")

        if cls.REQUEST_TIMEOUT <= 0:
            errors.append("REQUEST_TIMEOUT must be greater than zero.")

        if cls.MAX_RETRIES < 0:
            errors.append("MAX_RETRIES cannot be negative.")

        if cls.RETRY_BASE_SECONDS < 0:
            errors.append("RETRY_BASE_SECONDS cannot be negative.")

        if cls.MAX_OUTPUT_TOKENS <= 0:
            errors.append("MAX_OUTPUT_TOKENS must be greater than zero.")

        if len(cls.FALLBACK_MODELS) != len(set(cls.FALLBACK_MODELS)):
            errors.append("Duplicate fallback models detected.")

        if cls.OPENROUTER_MODEL in cls.FALLBACK_MODELS:
            errors.append("Primary model cannot also be a fallback model.")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(errors)
            )

        return True
