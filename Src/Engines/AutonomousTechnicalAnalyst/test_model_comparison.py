
import json
import re

from .ai_providers import ask_openrouter


TEST_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

TEST_PROMPT = """
You are a cryptocurrency technical analysis engine.

Analyze this hypothetical BTC-USDT scenario:

Timeframe: 4H
Market structure: Lower highs and lower lows
Price: Below EMA 50 and EMA 200
RSI 14: 38
ADX 14: 27
Recent price action: Bearish BOS
Nearby support: 65000
Nearby resistance: 67500

Return one JSON object with exactly these fields:

{
    "decision": "LONG, SHORT, or WAIT",
    "bias": "BULLISH, BEARISH, or NEUTRAL",
    "confidence": 0,
    "entry": 0,
    "stop_loss": 0,
    "take_profit": 0,
    "risk_reward": 0,
    "reason": "Short explanation",
    "invalidation": "Invalidation condition"
}

Rules:
- Do not invent missing market data.
- Confidence must be between 0 and 100.
- Choose WAIT if information is insufficient.
- Do not use Markdown code fences.
- Return JSON only.
- Keep the explanation concise.
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


def extract_json_text(response_text):
    text = response_text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def validate_json_response(response_text):
    cleaned_text = extract_json_text(response_text)

    try:
        parsed = json.loads(cleaned_text)

        if not isinstance(parsed, dict):
            return False, "Root value is not a JSON object.", None

        missing_keys = REQUIRED_KEYS - set(parsed.keys())

        if missing_keys:
            return (
                False,
                "Missing keys: " + ", ".join(sorted(missing_keys)),
                parsed,
            )

        if parsed["decision"] not in (
            "LONG",
            "SHORT",
            "WAIT",
        ):
            return False, "Invalid decision value.", parsed

        if parsed["bias"] not in (
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        ):
            return False, "Invalid bias value.", parsed

        confidence = parsed["confidence"]

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 100
        ):
            return False, "Invalid confidence value.", parsed

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
                return False, f"Invalid numeric field: {key}", parsed

        if not isinstance(parsed["reason"], str):
            return False, "Reason must be a string.", parsed

        if not isinstance(parsed["invalidation"], str):
            return False, "Invalid invalidation field.", parsed

        if parsed["decision"] == "SHORT":
            if parsed["stop_loss"] <= parsed["entry"]:
                return False, "SHORT stop loss must be above entry.", parsed

            if parsed["take_profit"] >= parsed["entry"]:
                return False, "SHORT take profit must be below entry.", parsed

        if parsed["decision"] == "LONG":
            if parsed["stop_loss"] >= parsed["entry"]:
                return False, "LONG stop loss must be below entry.", parsed

            if parsed["take_profit"] <= parsed["entry"]:
                return False, "LONG take profit must be above entry.", parsed

        if parsed["decision"] == "WAIT":
            if parsed["bias"] not in (
                "BULLISH",
                "BEARISH",
                "NEUTRAL",
            ):
                return False, "Invalid WAIT bias.", parsed

        return True, "JSON structure and basic risk checks passed.", parsed

    except json.JSONDecodeError as exc:
        return (
            False,
            f"JSON parsing error: {exc}",
            None,
        )


def main():
    print("=" * 70)
    print("AUTONOMOUS AI MODEL VALIDATION TEST")
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

        usage = result.get("usage", {})
        diagnostics = result.get("diagnostics", {})

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
        parsed = None

        if result["status"] == "success":
            response_text = result["response_text"]

            print("\nModel response:")
            print(response_text)

            valid_json, validation_message, parsed = (
                validate_json_response(response_text)
            )

        print("\nValidation:")
        print(f"Valid: {valid_json}")
        print(f"Details: {validation_message}")

        if parsed:
            print("\nParsed decision:")
            print(f"Decision: {parsed['decision']}")
            print(f"Bias: {parsed['bias']}")
            print(f"Confidence: {parsed['confidence']}")
            print(f"Entry: {parsed['entry']}")
            print(f"Stop loss: {parsed['stop_loss']}")
            print(f"Take profit: {parsed['take_profit']}")
            print(f"Risk reward: {parsed['risk_reward']}")

        results.append({
            "model": model,
            "status": result["status"],
            "latency": result["latency_seconds"],
            "valid_json": valid_json,
            "decision": (
                parsed["decision"]
                if parsed
                else "NONE"
            ),
            "error": result["error"],
        })

    print("\n" + "=" * 70)
    print("FINAL MODEL VALIDATION SUMMARY")
    print("=" * 70)

    for item in results:
        print("-" * 60)
        print(f"Model: {item['model']}")
        print(f"Status: {item['status']}")
        print(f"Latency: {item['latency']} seconds")
        print(f"Valid JSON: {item['valid_json']}")
        print(f"Decision: {item['decision']}")

        if item["error"]:
            print(f"Error: {item['error']}")

    valid_models = [
        item for item in results
        if item["status"] == "success" and item["valid_json"]
    ]

    print("\n" + "=" * 70)
    print(f"VALID MODELS: {len(valid_models)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
