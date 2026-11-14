
import json
import re
import time

from .ai_providers import ask_openrouter


TEST_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]


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


SCENARIOS = [
    {
        "id": "BULLISH_TREND",
        "expected": "LONG",
        "description": "Strong bullish trend",
        "market": {
            "symbol": "BTC-USDT",
            "timeframe": "4H",
            "current_price": 70000,
            "structure": "Higher highs and higher lows",
            "ema_50": 69000,
            "ema_200": 67500,
            "rsi_14": 62,
            "adx_14": 29,
            "recent_bos": "Bullish BOS",
            "support": 68000,
            "resistance": 71000,
        },
    },
    {
        "id": "BEARISH_TREND",
        "expected": "SHORT",
        "description": "Strong bearish trend",
        "market": {
            "symbol": "BTC-USDT",
            "timeframe": "4H",
            "current_price": 66500,
            "structure": "Lower highs and lower lows",
            "ema_50": 67000,
            "ema_200": 69000,
            "rsi_14": 38,
            "adx_14": 27,
            "recent_bos": "Bearish BOS",
            "support": 65000,
            "resistance": 67500,
        },
    },
    {
        "id": "SIDEWAYS_MARKET",
        "expected": "WAIT",
        "description": "Weak sideways market",
        "market": {
            "symbol": "BTC-USDT",
            "timeframe": "4H",
            "current_price": 66200,
            "structure": "Price moving inside a horizontal range",
            "ema_50": 66250,
            "ema_200": 66180,
            "rsi_14": 50,
            "adx_14": 14,
            "recent_bos": "No confirmed BOS",
            "support": 65000,
            "resistance": 67500,
        },
    },
    {
        "id": "CONFLICTING_SIGNALS",
        "expected": "WAIT",
        "description": "Conflicting market signals",
        "market": {
            "symbol": "BTC-USDT",
            "timeframe": "4H",
            "current_price": 68000,
            "structure": "Higher highs and higher lows",
            "ema_50": 68500,
            "ema_200": 69000,
            "rsi_14": 51,
            "adx_14": 25,
            "recent_bos": "Bullish BOS",
            "support": 67000,
            "resistance": 70000,
        },
    },
    {
        "id": "INCOMPLETE_DATA",
        "expected": "WAIT",
        "description": "Incomplete market data",
        "market": {
            "symbol": "BTC-USDT",
            "timeframe": "4H",
            "current_price": 68000,
            "structure": None,
            "ema_50": 67500,
            "ema_200": None,
            "rsi_14": None,
            "adx_14": None,
            "recent_bos": None,
            "support": 66000,
            "resistance": 70000,
        },
    },
]


def build_prompt(scenario):
    market_json = json.dumps(
        scenario["market"],
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are a professional cryptocurrency market analysis engine.

Analyze the following BTC-USDT market data.

MARKET DATA:
{market_json}

Return exactly one JSON object with these fields:

{{
    "decision": "LONG, SHORT, or WAIT",
    "bias": "BULLISH, BEARISH, or NEUTRAL",
    "confidence": 0,
    "entry": 0,
    "stop_loss": 0,
    "take_profit": 0,
    "risk_reward": 0,
    "reason": "Concise explanation",
    "invalidation": "Condition that invalidates the analysis"
}}

STRICT RULES:

1. Never invent missing market data.
2. Never assume an unknown indicator value.
3. Choose WAIT when data is incomplete or signals conflict.
4. Choose WAIT when there is no meaningful trading opportunity.
5. LONG requires stop_loss below entry and take_profit above entry.
6. SHORT requires stop_loss above entry and take_profit below entry.
7. Calculate risk_reward using actual entry, stop_loss and take_profit.
8. For WAIT, set entry, stop_loss, take_profit and risk_reward to zero.
9. Confidence must be between 0 and 100.
10. Do not use Markdown code fences.
11. Do not include text outside the JSON object.
12. Keep the reason and invalidation concise.

Return JSON only.
"""


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


def validate_response(response_text, scenario):
    cleaned = extract_json_text(response_text)

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "error": f"JSON parsing error: {exc}",
            "data": None,
            "checks": {},
            "score": 0,
        }

    if not isinstance(data, dict):
        return {
            "valid": False,
            "error": "Root value is not a JSON object.",
            "data": None,
            "checks": {},
            "score": 0,
        }

    missing = REQUIRED_KEYS - set(data.keys())

    if missing:
        return {
            "valid": False,
            "error": "Missing keys: " + ", ".join(sorted(missing)),
            "data": data,
            "checks": {},
            "score": 0,
        }

    checks = {}

    checks["decision"] = data["decision"] in (
        "LONG",
        "SHORT",
        "WAIT",
    )

    checks["bias"] = data["bias"] in (
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    )

    confidence = data["confidence"]

    checks["confidence"] = (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0 <= confidence <= 100
    )

    numeric_fields = (
        "entry",
        "stop_loss",
        "take_profit",
        "risk_reward",
    )

    checks["numeric_fields"] = all(
        isinstance(data[key], (int, float))
        and not isinstance(data[key], bool)
        and data[key] >= 0
        for key in numeric_fields
    )

    checks["text_fields"] = (
        isinstance(data["reason"], str)
        and bool(data["reason"].strip())
        and isinstance(data["invalidation"], str)
        and bool(data["invalidation"].strip())
    )

    if not all(checks.values()):
        return {
            "valid": False,
            "error": "One or more JSON field checks failed.",
            "data": data,
            "checks": checks,
            "score": 0,
        }

    decision = data["decision"]
    entry = data["entry"]
    stop = data["stop_loss"]
    target = data["take_profit"]
    reported_rr = data["risk_reward"]

    checks["risk_geometry"] = False
    checks["risk_reward_accuracy"] = False
    checks["entry_location"] = False

    if decision == "WAIT":
        checks["risk_geometry"] = (
            entry == 0
            and stop == 0
            and target == 0
            and reported_rr == 0
        )

        checks["risk_reward_accuracy"] = (
            reported_rr == 0
        )

        checks["entry_location"] = True

    elif decision == "LONG":
        checks["risk_geometry"] = (
            0 < stop < entry < target
        )

        market = scenario["market"]
        price = market["current_price"]
        support = market["support"]

        checks["entry_location"] = (
            support <= entry <= price
        )

        if checks["risk_geometry"]:
            actual_rr = (
                (target - entry)
                / (entry - stop)
            )

            checks["risk_reward_accuracy"] = (
                abs(actual_rr - reported_rr)
                <= max(0.05, actual_rr * 0.05)
            )

    elif decision == "SHORT":
        checks["risk_geometry"] = (
            0 < target < entry < stop
        )

        market = scenario["market"]
        price = market["current_price"]
        resistance = market["resistance"]

        checks["entry_location"] = (
            price <= entry <= resistance
        )

        if checks["risk_geometry"]:
            actual_rr = (
                (entry - target)
                / (stop - entry)
            )

            checks["risk_reward_accuracy"] = (
                abs(actual_rr - reported_rr)
                <= max(0.05, actual_rr * 0.05)
            )

    checks["decision_alignment"] = (
        decision == scenario["expected"]
    )

    score = 0

    if all(
        checks[key]
        for key in (
            "decision",
            "bias",
            "confidence",
            "numeric_fields",
            "text_fields",
        )
    ):
        score += 30

    if checks["decision_alignment"]:
        score += 25

    if checks["risk_geometry"]:
        score += 20

    if checks["risk_reward_accuracy"]:
        score += 15

    if checks["entry_location"]:
        score += 10

    return {
        "valid": True,
        "error": None,
        "data": data,
        "checks": checks,
        "score": score,
    }


def run_model_scenario(model, scenario):
    print("\n" + "-" * 70)
    print(f"MODEL: {model}")
    print(f"SCENARIO: {scenario['id']}")
    print(f"EXPECTED: {scenario['expected']}")
    print("-" * 70)

    started = time.time()

    result = ask_openrouter(
        build_prompt(scenario),
        model_override=model,
    )

    elapsed = round(time.time() - started, 3)

    item = {
        "model": model,
        "scenario": scenario["id"],
        "expected": scenario["expected"],
        "status": result["status"],
        "latency": elapsed,
        "decision": "NONE",
        "valid_json": False,
        "score": 0,
        "checks": {},
        "error": result["error"],
    }

    print(f"Provider status: {result['status']}")
    print(f"Latency: {elapsed} seconds")
    print(f"Attempts: {result['attempts']}")

    if result["status"] != "success":
        print(f"Provider error: {result['error']}")
        return item

    validation = validate_response(
        result["response_text"],
        scenario,
    )

    item["valid_json"] = validation["valid"]
    item["score"] = validation["score"]
    item["checks"] = validation["checks"]

    if not validation["valid"]:
        item["error"] = validation["error"]
        print(f"Validation error: {validation['error']}")
        return item

    data = validation["data"]

    item["decision"] = data["decision"]

    print("\nModel decision:")
    print(json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nValidation checks:")

    for name, passed in validation["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"\nScenario score: {validation['score']}/100")

    return item


def summarize_results(results):
    print("\n" + "=" * 80)
    print("MULTI-SCENARIO MODEL COMPARISON")
    print("=" * 80)

    model_summaries = []

    for model in TEST_MODELS:
        model_results = [
            item for item in results
            if item["model"] == model
        ]

        valid_count = sum(
            item["valid_json"]
            for item in model_results
        )

        aligned_count = sum(
            item["decision"] == item["expected"]
            for item in model_results
            if item["valid_json"]
        )

        total_score = sum(
            item["score"]
            for item in model_results
        )

        average_score = (
            total_score / len(model_results)
            if model_results
            else 0
        )

        successful_latencies = [
            item["latency"]
            for item in model_results
            if item["status"] == "success"
        ]

        average_latency = (
            sum(successful_latencies)
            / len(successful_latencies)
            if successful_latencies
            else 0
        )

        summary = {
            "model": model,
            "valid_responses": valid_count,
            "aligned_decisions": aligned_count,
            "average_score": round(average_score, 2),
            "average_latency": round(average_latency, 2),
        }

        model_summaries.append(summary)

        print("\n" + "-" * 70)
        print(f"Model: {model}")
        print(f"Valid JSON: {valid_count}/5")
        print(f"Expected decisions matched: {aligned_count}/5")
        print(f"Average score: {average_score:.2f}/100")
        print(f"Average latency: {average_latency:.2f} seconds")

    print("\n" + "=" * 80)
    print("SCENARIO DECISION MATRIX")
    print("=" * 80)

    for item in results:
        print(
            f"{item['model']} | "
            f"{item['scenario']} | "
            f"Expected: {item['expected']} | "
            f"Actual: {item['decision']} | "
            f"Score: {item['score']}"
        )

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

    return model_summaries


def main():
    print("=" * 80)
    print("AUTONOMOUS AI MULTI-SCENARIO TEST")
    print("=" * 80)

    print(f"Models: {len(TEST_MODELS)}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Total requests: {len(TEST_MODELS) * len(SCENARIOS)}")

    results = []

    for model in TEST_MODELS:
        for scenario in SCENARIOS:
            item = run_model_scenario(
                model,
                scenario,
            )

            results.append(item)

            time.sleep(1)

    summaries = summarize_results(results)

    with open(
        "multi_scenario_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "results": results,
                "summaries": summaries,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nResults saved to multi_scenario_results.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
