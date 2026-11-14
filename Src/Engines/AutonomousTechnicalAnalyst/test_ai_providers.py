
from .ai_providers import ask_openrouter
from .config import Config


TEST_PROMPT = (
    "Reply with exactly this text and nothing else: "
    "AI_PROVIDER_TEST_SUCCESS"
)


def main():
    print("=" * 60)
    print("AUTONOMOUS OPENROUTER ENGINE TEST")
    print("=" * 60)

    config = Config.validate()

    print("\nConfiguration status:")
    print(f"Valid: {config['valid']}")
    print(f"Models: {config['models']}")

    if not config["valid"]:
        for error in config["errors"]:
            print(f"Configuration error: {error}")

        return 1

    print("\nSending test request...")

    result = ask_openrouter(TEST_PROMPT)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Provider: {result['provider']}")
    print(f"Model: {result['model']}")
    print(f"Status: {result['status']}")
    print(f"Latency: {result['latency_seconds']} seconds")
    print(f"Attempts: {result['attempts']}")

    if result["status"] == "success":
        print(f"AI response: {result['response_text']}")
        print(f"Token usage: {result['usage']}")

        if result["response_text"] == "AI_PROVIDER_TEST_SUCCESS":
            print("\nRESULT: OPENROUTER TEST PASSED")
            return 0

        print("\nWARNING: AI response differs from expected text.")
        print("RESULT: CONNECTION PASSED, RESPONSE CHECK FAILED")
        return 1

    print(f"\nError: {result['error']}")
    print("\nRESULT: OPENROUTER TEST FAILED")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
