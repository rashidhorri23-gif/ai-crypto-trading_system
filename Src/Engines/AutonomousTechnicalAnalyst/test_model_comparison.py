
import json

from .ai_providers import ask_openrouter


TEST_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "baidu/ernie-5.5-flash-preview:free",
    "thinkingmachines/inkling-2-small:free",
]

TEST_PROMPT = """
You are a cryptocurrency technical analysis engine.

Analyze this hypothetical BTC-USDT market scenario:

- Timeframe: 4H
- Market structure: Lower highs and lower lows
- Price: Below EMA 50 and EMA 200
- RSI 14: 38
- ADX 14: 27
- Recent price action: Bearish BOS
- Nearby support: 65000
- Nearby resistance: 67500

Return only a valid JSON object using this exact structure:

{
    "decision": "LONG, SHORT, or WAIT",
    "bias": "BULLISH, BEARISH, or NEUTRAL",
    "confidence": 0,
    "entry": 0,
    "stop_loss": 0,
    "take_profit": 0,
    "risk_reward": 0,
    "reason": "Short explanation",
    "invalidation": "Condition that invalidates the analysis"
}

Rules:
- Do not invent missing market data.
- Confidence must be between 0 and 100.
- If there is insufficient information, choose WAIT.
- Return JSON only.
"""


REQUIRED_KEYS = {
    "decision",
    "bias",
    "confidence",
    "entry",
    "stop_loss",
    "take_profit",
    "risk_reward",
    "reason",
    "invalidation",
}


def validate_json_response(response_text):
    try:
        parsed = json.loads(response_text)

        if not isinstance(parsed, dict):
            return False, "Root value is not a JSON object."

        missing_keys = REQUIRED_KEYS - set(parsed.keys())

        if missing_keys:
            return False, (
                "Missing keys: "
                + ", ".join(sorted(missing_keys))
            )

        if parsed["decision"] not in (
            "LONG",
            "SHORT",
            "WAIT",
        ):
            return False, "Invalid decision value."

        if parsed["bias"] not in (
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        ):
            return False, "Invalid bias value."

        confidence = parsed["confidence"]

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 100
        ):
            return False, "Invalid confidence value."

        for key in (
            "entry",
            "stop_loss",
            "take_profit",
            "risk_reward",
        ):
            value = parsed[key]

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                return False, f"Invalid numeric field: {key}"

        if not isinstance(parsed["reason"], str):
            return False, "Reason must be a string."

        if not isinstance(parsed["invalidation"], str):
            return False, "Invalid invalidation field."

        return True, "All JSON fields are valid."

    except json.JSONDecodeError as exc:
        return False, f"JSON parsing error: {exc}"


def main():
    print("=" * 70)
    print("AUTONOMOUS AI MODEL DIAGNOSTIC TEST")
    print("=" * 70)

    results = []

    for model in TEST_MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)

        result = ask_openrouter(
            TEST_PROMPT,
            model_override=model,
        )

        diagnostics = result.get("diagnostics", {})
        usage = result.get("usage", {})

        print(f"Status: {result['status']}")
        print(f"Latency: {result['latency_seconds']} seconds")
        print(f"Attempts: {result['attempts']}")

        print("\nToken usage:")
        print(json.dumps(usage, indent=2))

        print("\nResponse diagnostics:")
        print(json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        ))

        valid_json = False
        validation_message = "No response available."

        if result["status"] == "success":
            response_text = result["response_text"]

            print("\nModel response:")
            print(response_text)

            valid_json, validation_message = (
                validate_json_response(response_text)
            )

        print("\nJSON validation:")
        print(f"Valid: {valid_json}")
        print(f"Details: {validation_message}")

        results.append({
            "model": model,
            "status": result["status"],
            "latency": result["latency_seconds"],
            "attempts": result["attempts"],
            "valid_json": valid_json,
            "finish_reason": diagnostics.get(
                "finish_reason",
                "unknown",
            ),
            "token_usage": usage,
            "error": result["error"],
        })

    print("\n" + "=" * 70)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 70)

    for item in results:
        print("-" * 60)
        print(f"Model: {item['model']}")
        print(f"Status: {item['status']}")
        print(f"Latency: {item['latency']} seconds")
        print(f"Attempts: {item['attempts']}")
        print(f"Finish reason: {item['finish_reason']}")
        print(f"Valid JSON: {item['valid_json']}")
        print(f"Token usage: {item['token_usage']}")

        if item["error"]:
            print(f"Error: {item['error']}")

    print("\nDIAGNOSTIC TEST COMPLETED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
