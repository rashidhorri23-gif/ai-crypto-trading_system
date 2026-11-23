import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from pathlib import Path

from .config import Config


BLOCKED_MODEL_IDS = {
    "thinkingmachines/inkling-2-small:free",
}

KNOWN_WORKING_MODEL = "nvidia/nemotron-3.5-super-vl:free"


def _read_error_body(error):
    try:
        return error.read().decode(
            "utf-8",
            errors="replace",
        )[:2000]
    except Exception:
        return str(error)




def _retry_after_seconds(error, fallback_seconds):
    """Read Retry-After as seconds or an HTTP date."""
    headers = getattr(error, "headers", None)
    retry_after = headers.get("Retry-After") if headers else None

    if not retry_after:
        return float(fallback_seconds)

    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (retry_at - datetime.now(timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return float(fallback_seconds)


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


def discover_free_vision_models(limit=20):
    """Discover free vision models that accept images and return text."""

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

            if not model_id:
                continue

            if model_id in BLOCKED_MODEL_IDS:
                continue

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

        discovered.sort(
            key=lambda item: (
                item["id"] != KNOWN_WORKING_MODEL
            )
        )

        discovered = discovered[:limit]

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

    discovered = discover_free_vision_models(limit=20)

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
        if preferred in BLOCKED_MODEL_IDS:
            return {
                "status": "error",
                "models": [],
                "error": (
                    "Configured vision model is blocked: "
                    + preferred
                ),
            }

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
    """Reject incomplete or invalid JSON responses."""

    cleaned = content.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            "AI returned incomplete or invalid JSON: "
            f"{error.msg} at line {error.lineno}, "
            f"column {error.colno}."
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "AI response JSON must be an object."
        )

    return True


def ask_openrouter_vision(
    image_path,
    prompt,
    model_override=None,
):
    """Send a local chart image and prompt to a vision model."""

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
        if model_override in BLOCKED_MODEL_IDS:
            return {
                "status": "error",
                "model": None,
                "content": None,
                "error": (
                    "Requested model is blocked: "
                    + model_override
                ),
            }

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

    model_errors = []

    for model_id in models:
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
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
            "max_tokens": Config.MAX_OUTPUT_TOKENS,
            "temperature": 0.1,
        }

        rate_limit_retried = False

        while True:
            try:
                response = _request_json(
                    url,
                    method="POST",
                    payload=payload,
                )

                choices = response.get("choices", [])

                if not choices:
                    raise ValueError(
                        "The AI response contains no choices."
                    )

                finish_reason = choices[0].get(
                    "finish_reason"
                )

                if finish_reason == "length":
                    model_error = (
                        f"Model {model_id}: output was truncated "
                        "at the model token limit; incomplete output rejected."
                    )
                    model_errors.append(model_error)
                    print("WARNING: " + model_error)
                    break

                if finish_reason == "content_filter":
                    model_error = (
                        f"Model {model_id}: output was stopped "
                        "by a content filter."
                    )
                    model_errors.append(model_error)
                    break

                content = _extract_text(response)

                if not content:
                    raise ValueError(
                        "The AI returned an empty response."
                    )

                _validate_json_response(content)

                return {
                    "status": "success",
                    "model": model_id,
                    "content": content,
                    "error": None,
                }

            except urllib.error.HTTPError as error:
                error_body = _read_error_body(error)

                if error.code == 429 and not rate_limit_retried:
                    retry_delay = _retry_after_seconds(
                        error,
                        Config.RETRY_BASE_SECONDS,
                    )
                    max_wait = getattr(
                        Config,
                        "RATE_LIMIT_MAX_WAIT_SECONDS",
                        30,
                    )

                    if retry_delay <= max_wait:
                        rate_limit_retried = True
                        print(
                            f"WARNING: Model {model_id} rate-limited "
                            f"(HTTP 429); retrying once after "
                            f"{retry_delay:.1f} seconds."
                        )
                        time.sleep(retry_delay)
                        continue

                    model_errors.append(
                        f"Model {model_id}: HTTP 429; Retry-After "
                        f"requested {retry_delay:.1f} seconds, exceeding "
                        f"the configured wait limit of {max_wait} seconds. "
                        "Skipping this model."
                    )
                    break

                model_errors.append(
                    f"Model {model_id}: HTTP {error.code}: {error_body}"
                )
                break

            except Exception as error:
                model_errors.append(
                    f"Model {model_id}: "
                    f"{type(error).__name__}: {error}"
                )
                break

    return {
        "status": "error",
        "model": None,
        "content": None,
        "error": "\n".join(model_errors)
        if model_errors
        else "All vision models failed.",
    }
