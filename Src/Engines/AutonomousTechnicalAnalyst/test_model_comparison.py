
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


def main():
    print("=" * 70)
    print("AUTONOMOUS AI MODEL COMPARISON TEST")
    print("=" * 70)

    results = []

    for model in TEST_MODELS:
        print("\n" + "-" * 70)
        print(f"Testing model: {model}")

        result = ask_openrouter(
            TEST_PROMPT,
            model_override=model,
        )

        print(f"Status: {result['status']}")
        print(f"Latency: {result['latency_seconds']} seconds")
        print(f"Attempts: {result['attempts']}")

        if result["status"] == "success":
            print("Response:")
            print(result["response_text"])

            try:
                parsed = json.loads(result["response_text"])

                required_keys = {
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

                valid_structure = required_keys.issubset(
                    parsed.keys()
                )

                print(
                    "JSON structure:",
                    "VALID" if valid_structure else "INVALID",
                )

            except (json.JSONDecodeError, TypeError):
                valid_structure = False
                print("JSON structure: INVALID")

        else:
            valid_structure = False
            print(f"Error: {result['error']}")

        results.append({
            "model": model,
            "status": result["status"],
            "latency": result["latency_seconds"],
            "valid_json": valid_structure,
            "response": result["response_text"],
            "error": result["error"],
        })

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    for item in results:
        print(f"\nModel: {item['model']}")
        print(f"Status: {item['status']}")
        print(f"Latency: {item['latency']} seconds")
        print(f"Valid JSON: {item['valid_json']}")

    print("\nCOMPARISON TEST COMPLETED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
