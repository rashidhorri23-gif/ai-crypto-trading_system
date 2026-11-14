
import os


class Config:
    """Configuration for the autonomous AI technical analyst."""

    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    ).strip()

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL",
        "baidu/ernie-5.5-flash-preview:free"
    ).strip()

    DEFAULT_FALLBACK_MODELS = (
        "nvidia/nemotron-3-ultra-550b-a55b:free,"
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    )

    FALLBACK_MODELS = [
        model.strip()
        for model in os.getenv(
            "OPENROUTER_FALLBACK_MODELS",
            DEFAULT_FALLBACK_MODELS
        ).split(",")
        if model.strip()
    ]

    MODEL_PRIORITY = list(
        dict.fromkeys(
            [OPENROUTER_MODEL] + FALLBACK_MODELS
        )
    )

    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 2
    RETRY_BASE_SECONDS = 2
    MAX_OUTPUT_TOKENS = 3000

    SYMBOL = "BTC_USDT"
    MARKET_TYPE = "SPOT"
    TIMEFRAME = "4H"

    DATA_DIRECTORY = "Data/Archive"

    @classmethod
    def validate(cls):
        errors = []

        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY is missing.")

        if not cls.OPENROUTER_MODEL:
            errors.append("OPENROUTER_MODEL is missing.")

        if cls.REQUEST_TIMEOUT <= 0:
            errors.append("REQUEST_TIMEOUT must be positive.")

        if cls.MAX_RETRIES < 0:
            errors.append("MAX_RETRIES cannot be negative.")

        if len(cls.MODEL_PRIORITY) != len(set(cls.MODEL_PRIORITY)):
            errors.append("Duplicate models found in model priority.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "models": cls.MODEL_PRIORITY,
        }
