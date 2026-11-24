import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import vision_provider


class VisionProviderValidationTests(unittest.TestCase):
    def test_invalid_analysis_schema_uses_next_model(self):
        responses = [
            {
                "model": "test/first-model:free",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"decision": "WAIT"})},
                }],
            },
            {
                "model": "test/second-model:free",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({
                            "decision": "WAIT",
                            "schema_version": "2.0",
                        })
                    },
                }],
            },
        ]

        def require_schema_version(candidate):
            if candidate.get("schema_version") != "2.0":
                raise ValueError("Missing required schema_version.")

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "chart.png"
            image_path.write_bytes(b"test-image")

            with (
                patch.object(vision_provider.Config, "OPENROUTER_API_KEY", "test-key"),
                patch.object(
                    vision_provider,
                    "build_vision_model_priority",
                    return_value={
                        "status": "success",
                        "models": [
                            {"id": "test/first-model:free", "supported_parameters": ["response_format"]},
                            {"id": "test/second-model:free", "supported_parameters": ["response_format"]},
                        ],
                        "error": None,
                    },
                ),
                patch.object(
                    vision_provider,
                    "_request_json",
                    side_effect=responses,
                ) as request_mock,
            ):
                result = vision_provider.ask_openrouter_vision(
                    image_path=image_path,
                    prompt="Return a JSON object.",
                    response_validator=require_schema_version,
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], "test/second-model:free")
        self.assertEqual(request_mock.call_count, 2)

    def test_truncated_output_uses_next_model(self):
        responses = [
            {
                "model": "test/first-model:free",
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": ""},
                }],
            },
            {
                "model": "test/second-model:free",
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"ok": True})},
                }],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "chart.png"
            image_path.write_bytes(b"test-image")

            with (
                patch.object(vision_provider.Config, "OPENROUTER_API_KEY", "test-key"),
                patch.object(
                    vision_provider,
                    "build_vision_model_priority",
                    return_value={
                        "status": "success",
                        "models": [
                            {"id": "test/first-model:free", "supported_parameters": ["response_format"]},
                            {"id": "test/second-model:free", "supported_parameters": ["response_format"]},
                        ],
                        "error": None,
                    },
                ),
                patch.object(
                    vision_provider,
                    "_request_json",
                    side_effect=responses,
                ) as request_mock,
            ):
                result = vision_provider.ask_openrouter_vision(
                    image_path=image_path,
                    prompt="Return a JSON object.",
                )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], "test/second-model:free")
        self.assertEqual(request_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
