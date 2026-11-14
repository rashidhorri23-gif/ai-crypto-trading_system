
import os


class Config:
    """Configuration for the autonomous AI technical analyst."""

    # OpenRouter connection
    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    ).strip()

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Primary model
    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free"
    ).strip()

    # Optional fallback models, separated by commas.
    # Example: model-one,model-two
    FALLBACK_MODELS = [
        model.strip()
        for model in os.getenv(
            "OPENROUTER_FALLBACK_MODELS",
            ""
        ).split(",")
        if model.strip()
    ]

    # Remove duplicates while preserving order.
    MODEL_PRIORITY = list(
        dict.fromkeys(
            [OPENROUTER_MODEL] + FALLBACK_MODELS
        )
    )

    # Request settings
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 2
    RETRY_BASE_SECONDS = 2
    MAX_OUTPUT_TOKENS = 3000

    # Initial market configuration
    SYMBOL = "BTC_USDT"
    MARKET_TYPE = "SPOT"
    TIMEFRAME = "4H"

    # Project paths
    DATA_DIRECTORY = "Data/Archive"

    @classmethod
    def validate(cls):
        """Validate the required configuration."""

        errors = []

        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY is missing.")

        if not cls.OPENROUTER_MODEL:
            errors.append("OPENROUTER_MODEL is missing.")

        if cls.REQUEST_TIMEOUT <= 0:
            errors.append("REQUEST_TIMEOUT must be positive.")

        if cls.MAX_RETRIES < 0:
            errors.append("MAX_RETRIES cannot be negative.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "models": cls.MODEL_PRIORITY,
        }
