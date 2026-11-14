
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import Config


def _make_result(provider, model, started_at, text="", usage=None, error=None):
    return {
        "provider": provider,
        "model": model,
        "response_text": text,
        "latency_seconds": round(time.time() - started_at, 3),
        "usage": usage or {},
        "status": "success" if not error else "error",
        "error": error,
    }


def _send_request(url, headers, payload):
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=Config.REQUEST_TIMEOUT
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_openrouter_text(data):
    choices = data.get("choices", [])

    if not choices:
        raise ValueError("OpenRouter returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))

        return "\n".join(parts).strip()

    raise ValueError("OpenRouter returned no readable text content.")


def _extract_gemini_text(data):
    candidates = data.get("candidates", [])

    if not candidates:
        raise ValueError("Gemini returned no candidates.")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])

    texts = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ]

    result = "\n".join(texts).strip()

    if not result:
        raise ValueError("Gemini returned no readable text content.")

    return result


def ask_openrouter(prompt):
    started_at = time.time()
    model = Config.OPENROUTER_MODEL

    if not Config.OPENROUTER_API_KEY:
        return _make_result(
            "openrouter", model, started_at,
            error="OPENROUTER_API_KEY is not configured."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 1200,
        "reasoning": {
            "effort": "none"
        },
    }

    try:
        data = _send_request(url, headers, payload)
        response_text = _extract_openrouter_text(data)

        return _make_result(
            "openrouter",
            model,
            started_at,
            text=response_text,
            usage=data.get("usage", {}),
        )

    except urllib.error.HTTPError as exc:
        return _make_result(
            "openrouter", model, started_at,
            error=f"HTTP error: {exc.code}"
        )

    except Exception as exc:
        return _make_result(
            "openrouter", model, started_at,
            error=f"{type(exc).__name__}: {exc}"
        )


def ask_gemini(prompt):
    started_at = time.time()
    model = Config.GEMINI_MODEL

    if not Config.GEMINI_API_KEY:
        return _make_result(
            "gemini", model, started_at,
            error="GEMINI_API_KEY is not configured."
        )

    encoded_model = urllib.parse.quote(model, safe="")
    encoded_key = urllib.parse.quote(Config.GEMINI_API_KEY, safe="")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{encoded_model}:generateContent?key={encoded_key}"
    )

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 1200,
            "temperature": 0.2,
        },
    }

    try:
        data = _send_request(url, headers, payload)
        response_text = _extract_gemini_text(data)

        return _make_result(
            "gemini",
            model,
            started_at,
            text=response_text,
            usage=data.get("usageMetadata", {}),
        )

    except urllib.error.HTTPError as exc:
        return _make_result(
            "gemini", model, started_at,
            error=f"HTTP error: {exc.code}"
        )

    except Exception as exc:
        return _make_result(
            "gemini", model, started_at,
            error=f"{type(exc).__name__}: {exc}"
        )

