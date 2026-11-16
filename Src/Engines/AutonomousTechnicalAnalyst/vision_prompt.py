
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path

from .config import Config


def _read_error_body(error):
    try:
        return error.read().decode(
            "utf-8",
            errors="replace",
        )[:2000]
    except Exception:
        return str(error)


def _request_json(url, method="GET", payload=None):
    headers = {
        "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    body = None

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        headers=headers,
        data=body,
        method=method,
    )

    with urllib.request.urlopen(
        request,
        timeout=Config.REQUEST_TIMEOUT,
    ) as response:
        return json.loads(
            response.read().decode("utf-8")
        )


def discover_free_vision_models(limit=10):
    """Discover free models that accept images and return text."""

    if not Config.OPENROUTER_API_KEY:
        return {
            "status": "error",
            "models": [],
            "error": "OPENROUTER_API_KEY is not configured.",
        }

    query = urllib.parse.urlencode({
        "input_modalities": "image",
        "sort": "most-popular",
    })

    url = (
        Config.OPENROUTER_BASE_URL
        + "/models?"
        + query
    )

    try:
        data = _request_json(url)

        discovered = []

        for item in data.get("data", []):
            model_id = item.get("id", "")
            architecture = item.get("architecture", {})
            pricing = item.get("pricing", {})

            input_modalities = architecture.get(
                "input_modalities", []
            )

            output_modalities = architecture.get(
                "output_modalities", []
            )

            try:
                prompt_price = float(
                    pricing.get("prompt", -1)
                )

                completion_price = float(
                    pricing.get("completion", -1)
                )
            except (TypeError, ValueError):
                continue

            if not model_id.endswith(":free"):
                continue

            if prompt_price != 0 or completion_price != 0:
                continue

            if "image" not in input_modalities:
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
            })

            if len(discovered) >= limit:
                break

        return {
            "status": "success",
            "models": discovered,
            "error": None,
        }

    except urllib.error.HTTPError as error:
        return {
            "status": "error",
            "models": [],
            "error": (
                f"HTTP {error.code}: "
                f"{_read_error_body(error)}"
            ),
        }

    except Exception as error:
        return {
            "status": "error",
            "models": [],
            "error": f"{type(error).__name__}: {error}",
        }


def build_vision_model_priority():
    """Build the priority list for vision-capable models."""

    discovered = discover_free_vision_models(limit=10)

    if discovered["status"] != "success":
        return {
            "status": "error",
            "models": [],
            "error": discovered["error"],
        }

    models = [
        item["id"]
        for item in discovered["models"]
    ]

    preferred = os.getenv(
        "OPENROUTER_VISION_MODEL",
        "",
    ).strip()

    if preferred:
        if preferred not in models:
            return {
                "status": "error",
                "models": [],
                "error": (
                    "Configured vision model is not "
                    "available in the verified free vision list: "
                    + preferred
                ),
            }

        models.remove(preferred)
        models.insert(0, preferred)

    return {
        "status": "success",
        "models": models,
        "error": None,
    }


def _extract_text(response):
    choices = response.get("choices", [])

    if not choices:
        raise ValueError(
            "The AI response contains no choices."
        )

    message = choices[0].get("message", {})
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
            ):
                text_parts.append(
                    part.get("text", "")
                )

        return "\n".join(text_parts).strip()

    raise ValueError(
        "The AI response does not contain readable text."
    )


def _validate_json_response(content):
    """Check whether the model returned a complete JSON object."""

    try:
        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            return False, "Response is not a JSON object."

        return True, None

    except json.JSONDecodeError as error:
        return (
            False,
            (
                f"Invalid JSON: {error.msg} "
                f"at line {error.lineno}, "
                f"column {error.colno}."
            ),
        )


def _build_retry_prompt(original_prompt, attempt):
    """Build a stricter prompt for retry attempts."""

    return (
        original_prompt
        + "\n\n"
        + "IMPORTANT RETRY INSTRUCTIONS:\n"
        + "Your previous response was incomplete or invalid.\n"
        + "Return one complete JSON object only.\n"
        + "Do not use Markdown or code fences.\n"
        + "Keep descriptions concise.\n"
        + "Do not omit required JSON keys.\n"
        + "Do not stop before closing every JSON object and string.\n"
        + "Use null for unavailable numeric values.\n"
        + f"Retry attempt: {attempt}\n"
    )


def ask_openrouter_vision(
    image_path,
    prompt,
    model_override=None,
):
    """Send a chart image and prompt to a vision model."""

    if not Config.OPENROUTER_API_KEY:
        return {
            "status": "error",
            "model": None,
            "content": None,
            "error": "OPENROUTER_API_KEY is not configured.",
        }

    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "status": "error",
            "model": None,
            "content": None,
            "error": "Prompt must be a non-empty string.",
        }

    image_file = Path(image_path)

    if not image_file.is_file():
        return {
            "status": "error",
            "model": None,
            "content": None,
            "error": f"Image file not found: {image_path}",
        }

    mime_type, _ = mimetypes.guess_type(
        str(image_file)
    )

    if mime_type not in (
        "image/png",
        "image/jpeg",
        "image/webp",
    ):
        return {
            "status": "error",
            "model": None,
            "content": None,
            "error": "Only PNG, JPEG and WEBP images are supported.",
        }

    try:
        image_bytes = image_file.read_bytes()

        if not image_bytes:
            raise ValueError("Image file is empty.")

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("ascii")

        image_data_url = (
            f"data:{mime_type};base64,{encoded_image}"
        )

    except Exception as error:
        return {
            "status": "error",
            "model": None,
            "content": None,
            "error": f"{type(error).__name__}: {error}",
        }

    if model_override:
        models = [model_override]
    else:
        priority = build_vision_model_priority()

        if priority["status"] != "success":
            return {
                "status": "error",
                "model": None,
                "content": None,
                "error": priority["error"],
            }

        models = priority["models"]

    if not models:
        return {
            "status": "error",
            "model": None,
            "content": None,
            "error": "No vision-capable models are available.",
        }

    url = (
        Config.OPENROUTER_BASE_URL
        + "/chat/completions"
    )

    last_error = None
    last_content = None
    total_attempts = Config.MAX_RETRIES + 1

    for model_id in models:

        for attempt in range(total_attempts):

            max_tokens = min(
                Config.MAX_OUTPUT_TOKENS * (attempt + 1),
                16000,
            )

            current_prompt = prompt

            if attempt > 0:
                current_prompt = _build_retry_prompt(
                    prompt,
                    attempt,
                )

            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": current_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url,
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            try:
                response = _request_json(
                    url,
                    method="POST",
                    payload=payload,
                )

                content = _extract_text(response)
                last_content = content

                if not content:
                    raise ValueError(
                        "The AI returned an empty response."
                    )

                choices = response.get("choices", [])
                finish_reason = choices[0].get(
                    "finish_reason"
                )

                if finish_reason == "length":
                    last_error = (
                        f"Model {model_id} reached its "
                        "output token limit."
                    )

                    if attempt < total_attempts - 1:
                        time.sleep(
                            Config.RETRY_BASE_SECONDS
                        )
                        continue

                    break

                is_valid, validation_error = (
                    _validate_json_response(content)
                )

                if not is_valid:
                    last_error = (
                        f"Model {model_id}: "
                        f"{validation_error}"
                    )

                    if attempt < total_attempts - 1:
                        time.sleep(
                            Config.RETRY_BASE_SECONDS
                        )
                        continue

                    break

                return {
                    "status": "success",
                    "model": model_id,
                    "content": content,
                    "finish_reason": finish_reason,
                    "error": None,
                }

            except urllib.error.HTTPError as error:
                error_body = _read_error_body(error)

                last_error = (
                    f"Model {model_id}: "
                    f"HTTP {error.code}: {error_body}"
                )

                if error.code not in (
                    400, 404, 408, 429,
                    500, 502, 503, 504,
                ):
                    break

                if attempt < total_attempts - 1:
                    time.sleep(
                        Config.RETRY_BASE_SECONDS
                    )

            except Exception as error:
                last_error = (
                    f"Model {model_id}: "
                    f"{type(error).__name__}: {error}"
                )

                if attempt < total_attempts - 1:
                    time.sleep(
                        Config.RETRY_BASE_SECONDS
                    )

        if last_content:
            is_valid, _ = _validate_json_response(
                last_content
            )

            if not is_valid:
                continue

    return {
        "status": "error",
        "model": None,
        "content": last_content,
        "error": (
            last_error
            or "All vision models failed."
        ),
    }
