
import json
import time
import urllib.error
import urllib.request

from .config import Config


RETRYABLE_HTTP_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


def _make_result(
    status,
    response_text="",
    model="",
    latency_seconds=0.0,
    usage=None,
    attempts=0,
    error=None,
):
    return {
        "status": status,
        "response_text": response_text,
        "model": model,
        "latency_seconds": round(latency_seconds, 3),
        "usage": usage or {},
        "attempts": attempts,
        "error": error,
    }


def _read_error_body(error):
    try:
        body = error.read().decode("utf-8", errors="replace")
        if not body:
            return str(error)

        try:
            parsed = json.loads(body)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            return body

    except Exception:
        return str(error)


def _send_request(payload):
    url = f"{Config.OPENROUTER_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rashidhorri23-gif/ai-crypto-trading_system",
        "X-Title": "Autonomous Crypto Technical Analyst",
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=Config.REQUEST_TIMEOUT,
    ) as response:
        response_body = response.read().decode("utf-8")

    return json.loads(response_body)


def _extract_response(data, model, latency_seconds, attempts):
    try:
        choices = data.get("choices", [])

        if not choices:
            return _make_result(
                status="error",
                model=model,
                latency_seconds=latency_seconds,
                attempts=attempts,
                error="No choices returned by OpenRouter.",
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        finish_reason = choice.get("finish_reason")

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    item_text = item.get("text")
                    if item_text:
                        text_parts.append(item_text)

            content = "\n".join(text_parts)

        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        usage = data.get("usage", {})

        if finish_reason == "length":
            return _make_result(
                status="error",
                model=model,
                latency_seconds=latency_seconds,
                usage=usage,
                attempts=attempts,
                error="Model output was truncated because the token limit was reached.",
            )

        if not content:
            return _make_result(
                status="error",
                model=model,
                latency_seconds=latency_seconds,
                usage=usage,
                attempts=attempts,
                error="OpenRouter returned empty response content.",
            )

        return _make_result(
            status="success",
            response_text=content,
            model=data.get("model", model),
            latency_seconds=latency_seconds,
            usage=usage,
            attempts=attempts,
        )

    except Exception as error:
        return _make_result(
            status="error",
            model=model,
            latency_seconds=latency_seconds,
            attempts=attempts,
            error=f"Response parsing failed: {str(error)}",
        )


def _request_with_retry(payload, model):
    last_error = None
    start_time = time.time()

    for attempt in range(1, Config.MAX_RETRIES + 2):
        try:
            data = _send_request(payload)

            latency = time.time() - start_time

            return _extract_response(
                data=data,
                model=model,
                latency_seconds=latency,
                attempts=attempt,
            )

        except urllib.error.HTTPError as error:
            error_body = _read_error_body(error)

            last_error = (
                f"HTTP {error.code}: {error_body}"
            )

            if error.code not in RETRYABLE_HTTP_CODES:
                break

        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as error:
            last_error = str(error)

        except Exception as error:
            last_error = str(error)
            break

        if attempt <= Config.MAX_RETRIES:
            delay = (
                Config.RETRY_BASE_SECONDS
                * (2 ** (attempt - 1))
            )
            time.sleep(delay)

    latency = time.time() - start_time

    return _make_result(
        status="error",
        model=model,
        latency_seconds=latency,
        attempts=Config.MAX_RETRIES + 1,
        error=last_error or "OpenRouter request failed.",
    )


def ask_openrouter(prompt, model_override=None):
    if not Config.OPENROUTER_API_KEY:
        return _make_result(
            status="error",
            error="OPENROUTER_API_KEY is missing.",
        )

    model = model_override or Config.OPENROUTER_MODEL

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "reasoning": {
            "effort": "low",
        },
        "max_tokens": Config.MAX_OUTPUT_TOKENS,
    }

    return _request_with_retry(
        payload=payload,
        model=model,
    )
