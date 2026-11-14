
from .ai_providers import ask_gemini, ask_openrouter


TEST_PROMPT = (
    "Reply with exactly this text and nothing else: "
    "AI_PROVIDER_TEST_SUCCESS"
)


def test_provider(provider_name, provider_function):
    print(f"\nTesting provider: {provider_name}")
    result = provider_function(TEST_PROMPT)

    print(f"Status: {result['status']}")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_seconds']} seconds")

    if result["status"] == "success":
        print(f"Response: {result['response_text']}")
        print("Provider test: PASSED")
        return True

    print(f"Error: {result['error']}")
    print("Provider test: FAILED")
    return False


def main():
    print("=" * 50)
    print("AUTONOMOUS AI PROVIDER TEST")
    print("=" * 50)

    gemini_ok = test_provider("Gemini", ask_gemini)
    openrouter_ok = test_provider("OpenRouter", ask_openrouter)

    print("\n" + "=" * 50)

    if gemini_ok and openrouter_ok:
        print("RESULT: BOTH PROVIDERS PASSED")
        return 0

    print("RESULT: ONE OR MORE PROVIDERS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
