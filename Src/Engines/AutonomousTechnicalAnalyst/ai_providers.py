
import json
import socket
import time
import urllib.error
import urllib.request

from .config import Config


def _make_result(
    provider,
    model,
    started_at,
    text="",
    usage=None,
    error=None,
    attempts=0,
    diagnostics=None,
):
    return {
        "provider": provider,
        "model": model,
        "response_text": text,
        "latency_seconds": round(time.time() - started_at, 3),
        "usage": usage or {},
        "status": "success" if not error else "error",
        "error": error,
        "attempts": attempts,
        "diagnostics": diagnostics or {},
    }


def _read_error_body(exc):
    try:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        if not body:
            return "No error details returned."

        try:
            data = json.loads(body)
            return json.dumps(
                data,
                ensure_ascii=False,
            )[:1500]

        except json.JSONDecodeError:
            return body[:1500]

    except Exception:
        return "Unable to read error response."


def _send_request(url, headers, payload):
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
        body = response.read().decode("utf-8")
        return json.loads(body)


def _extract_response(data):
    choices = data.get("choices", [])

    diagnostics = {
        "choice_count": len(choices),
        "finish_reason": None,
        "message_keys": [],
        "content_type": None,
        "content_length": 0,
        "content_preview": "",
        "reasoning_present": False,
        "reasoning_length": 0,
        "refusal": None,
        "tool_calls_count": 0,
    }

    if not choices:
        raise ValueError(
            "Model returned no choices. "
            + json.dumps(diagnostics)
        )

    choice = choices[0]
    message = choice.get("message", {})

    diagnostics["finish_reason"] = choice.get(
        "finish_reason"
    )

    diagnostics["message_keys"] = list(
        message.keys()
    )

    diagnostics["refusal"] = message.get("refusal")

    reasoning = message.get("reasoning")

    if isinstance(reasoning, str):
        diagnostics["reasoning_present"] = bool(
            reasoning.strip()
        )
        diagnostics["reasoning_length"] = len(reasoning)

    tool_calls = message.get("tool_calls", [])

    if isinstance(tool_calls, list):
        diagnostics["tool_calls_count"] = len(
            tool_calls
        )

    content = message.get("content")

    diagnostics["content_type"] = type(
        content
    ).__name__

    parts = []

    if isinstance(content, str):
        if content.strip():
            parts.append(content.strip())

    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    value = item.get("text", "")

                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())

    result_text = "\n".join(parts).strip()

    diagnostics["content_length"] = len(result_text)
    diagnostics["content_preview"] = result_text[:500]

    if not result_text:
        raise ValueError(
            "Model returned no readable text content. "
            "Diagnostics: "
            + json.dumps(
                diagnostics,
                ensure_ascii=False,
            )
        )

    return result_text, diagnostics


def _request_with_retry(url, headers, payload):
    max_attempts = Config.MAX_RETRIES + 1
    last_error = "Unknown request error."

    for attempt in range(max_attempts):
        try:
            data = _send_request(
                url,
                headers,
                payload,
            )

            return {
                "success": True,
                "data": data,
                "attempts": attempt + 1,
                "error": None,
            }

        except urllib.error.HTTPError as exc:
            error_details = _read_error_body(exc)

            last_error = (
                f"HTTP {exc.code}: {error_details}"
            )

            retryable = exc.code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

            if not retryable or attempt >= max_attempts - 1:
                return {
                    "success": False,
                    "data": None,
                    "attempts": attempt + 1,
                    "error": last_error,
                }

            retry_after = exc.headers.get("Retry-After")

            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = (
                    Config.RETRY_BASE_SECONDS
                    * (2 ** attempt)
                )

            time.sleep(min(max(delay, 1), 30))

        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            last_error = (
                f"Connection error: {exc}"
            )

            if attempt >= max_attempts - 1:
                return {
                    "success": False,
                    "data": None,
                    "attempts": attempt + 1,
                    "error": last_error,
                }

            delay = (
                Config.RETRY_BASE_SECONDS
                * (2 ** attempt)
            )

            time.sleep(min(delay, 30))

        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "attempts": attempt + 1,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    return {
        "success": False,
        "data": None,
        "attempts": max_attempts,
        "error": last_error,
    }


def ask_openrouter(prompt, model_override=None):
    started_at = time.time()

    config_status = Config.validate()

    if not config_status["valid"]:
        return _make_result(
            "openrouter",
            "",
            started_at,
            error="Configuration error: "
            + "; ".join(config_status["errors"]),
        )

    if model_override is not None:
        if (
            not isinstance(model_override, str)
            or not model_override.strip()
        ):
            return _make_result(
                "openrouter",
                "",
                started_at,
                error="Invalid model_override value.",
            )

    url = (
        Config.OPENROUTER_BASE_URL
        + "/chat/completions"
    )

    headers = {
        "Authorization": (
            f"Bearer {Config.OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
        "HTTP-Referer": (
            "https://github.com/"
            "rashidhorri23-gif/ai-crypto-trading_system"
        ),
        "X-Title": "Autonomous Crypto Technical Analyst",
    }

    models_to_try = (
        [model_override.strip()]
        if model_override
        else Config.MODEL_PRIORITY
    )

    error_history = []
    total_attempts = 0

    for model in models_to_try:
        print(f"Trying OpenRouter model: {model}")

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": Config.MAX_OUTPUT_TOKENS,
        }

        result = _request_with_retry(
            url,
            headers,
            payload,
        )

        total_attempts += result["attempts"]

        if not result["success"]:
            error_history.append(
                f"{model}: {result['error']}"
            )
            continue

        data = result["data"]

        usage = data.get("usage", {})

        try:
            response_text, diagnostics = _extract_response(
                data
            )

            return _make_result(
                "openrouter",
                model,
                started_at,
                text=response_text,
                usage=usage,
                attempts=total_attempts,
                diagnostics=diagnostics,
            )

        except Exception as exc:
            choices = data.get("choices", [])
            finish_reason = None

            if choices:
                finish_reason = choices[0].get(
                    "finish_reason"
                )

            error_history.append(
                f"{model}: Response validation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return _make_result(
                "openrouter",
                model,
                started_at,
                usage=usage,
                error=" | ".join(error_history),
                attempts=total_attempts,
                diagnostics={
                    "finish_reason": finish_reason,
                    "choice_count": len(choices),
                    "usage": usage,
                    "response_error": str(exc)[:1500],
                },
            )

    return _make_result(
        "openrouter",
        model_override or "all_models_failed",
        started_at,
        error=" | ".join(error_history),
        attempts=total_attempts,
    )
