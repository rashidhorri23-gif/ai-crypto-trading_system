import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from . import vision_provider
from .daily_btc_analysis import compact_vision_context


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

    def test_shared_free_daily_quota_stops_model_fallback_immediately(self):
        error_body = json.dumps({
            "error": {
                "message": "Rate limit exceeded: free-models-per-day.",
                "code": 429,
                "metadata": {
                    "limit_source": "openrouter_free_tier_daily",
                    "headers": {
                        "X-RateLimit-Remaining": "0",
                    },
                },
            }
        }).encode("utf-8")
        quota_error = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(error_body),
        )

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
                            {"id": "test/first-model:free", "supported_parameters": []},
                            {"id": "test/second-model:free", "supported_parameters": []},
                            {"id": "test/third-model:free", "supported_parameters": []},
                        ],
                        "error": None,
                    },
                ),
                patch.object(
                    vision_provider,
                    "_request_json",
                    side_effect=quota_error,
                ) as request_mock,
                patch.object(vision_provider.time, "sleep") as sleep_mock,
            ):
                result = vision_provider.ask_openrouter_vision(
                    image_path=image_path,
                    prompt="Return a JSON object.",
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("account-wide free-model daily quota is exhausted", result["error"])
        self.assertEqual(request_mock.call_count, 1)
        sleep_mock.assert_not_called()


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


class VisionContextCompressionTests(unittest.TestCase):
    def test_context_keeps_key_evidence_and_limits_candle_history(self):
        candles = [
            {
                "timestamp_utc": f"2026-11-24T{i:02d}:00:00+00:00",
                "open": 100 + i,
                "high": 102 + i,
                "low": 99 + i,
                "close": 101 + i,
                "rsi_14": 55,
                "ema_20": 110,
                "market_structure": "HH",
                "unused_large_field": "x" * 1000,
            }
            for i in range(20)
        ]
        context = {
            "symbol": "BTC-USDT",
            "source": "OKX",
            "analysis_time_utc": "2026-11-24T08:00:00+00:00",
            "timeframes": {
                timeframe: {
                    "source": "OKX",
                    "freshness": {"age_minutes": 2},
                    "data_quality": {"missing_interval_count": 0},
                    "closed_candle_count": 100,
                    "latest_close": 120,
                    "latest_confirmed_candle": candles[-1],
                    "recent_confirmed_candles": candles,
                    "detected_chart_patterns": [{
                        "name": "Ascending Triangle",
                        "direction": "BULLISH",
                        "status": "DEVELOPING",
                        "confidence": 70,
                        "target": 130,
                        "invalidation": 110,
                        "trigger": "شکست مقاومت",
                        "points": [{"time": 1, "price": 2}] * 100,
                        "lines": [{"points": [{"time": 1, "value": 2}] * 100}],
                    }],
                }
                for timeframe in ("4H", "15M", "5M")
            },
            "data_status": {},
            "data_quality": {},
        }

        compact = compact_vision_context(context, None, None)
        self.assertEqual(len(compact["timeframes"]["4H"]["recent_confirmed_candles"]), 8)
        latest = compact["timeframes"]["4H"]["latest_confirmed_candle"]
        self.assertEqual(latest["close"], 120)
        self.assertEqual(latest["market_structure"], "HH")
        self.assertNotIn("unused_large_field", latest)

        pattern = compact["timeframes"]["4H"]["detected_chart_patterns"][0]
        self.assertEqual(pattern["name"], "Ascending Triangle")
        self.assertNotIn("points", pattern)
        self.assertNotIn("lines", pattern)

        encoded_size = len(json.dumps(compact, ensure_ascii=False))
        self.assertLess(encoded_size, 18000)




if __name__ == "__main__":
    unittest.main()
