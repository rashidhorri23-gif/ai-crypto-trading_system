
import json
import urllib.error
import urllib.parse
import urllib.request

from .config import Config


def discover_free_chat_models(limit=5):
    """Discover currently listed free text chat models."""

    if not Config.OPENROUTER_API_KEY:
        return {
            "status": "error",
            "models": [],
            "error": "OPENROUTER_API_KEY is not configured.",
        }

    query = urllib.parse.urlencode({
        "endpoints": "/api/v1/chat/completions",
        "sort": "most-popular",
    })

    url = (
        Config.OPENROUTER_BASE_URL
        + "/models?"
        + query
    )

    headers = {
        "Authorization": (
            f"Bearer {Config.OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    request = urllib.request.Request(
        url=url,
        headers=headers,
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=Config.REQUEST_TIMEOUT,
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        discovered = []

        for item in data.get("data", []):
            model_id = item.get("id", "")
            pricing = item.get("pricing", {})
            architecture = item.get("architecture", {})

            input_modalities = architecture.get(
                "input_modalities", []
            )

            output_modalities = architecture.get(
                "output_modalities", []
            )

            prompt_price = pricing.get("prompt")
            completion_price = pricing.get("completion")

            if not model_id.endswith(":free"):
                continue

            if prompt_price != "0" or completion_price != "0":
                continue

            if "text" not in input_modalities:
                continue

            if "text" not in output_modalities:
                continue

            discovered.append({
                "id": model_id,
                "name": item.get("name", model_id),
                "context_length": item.get(
                    "context_length", 0
                ),
                "pricing": pricing,
            })

            if len(discovered) >= limit:
                break

        return {
            "status": "success",
            "models": discovered,
            "error": None,
        }

    except urllib.error.HTTPError as exc:
        return {
            "status": "error",
            "models": [],
            "error": (
                f"HTTP {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')[:1000]}"
            ),
        }

    except Exception as exc:
        return {
            "status": "error",
            "models": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_model_priority():
    """Build a model list with the configured primary model first."""

    primary = Config.OPENROUTER_MODEL
    result = [primary]

    discovered = discover_free_chat_models(limit=10)

    if discovered["status"] == "success":
        for model in discovered["models"]:
            model_id = model["id"]

            if model_id not in result:
                result.append(model_id)

    for model_id in Config.FALLBACK_MODELS:
        if model_id not in result:
            result.append(model_id)

    return {
        "status": "success",
        "models": result,
        "discovery_error": discovered["error"],
    }
