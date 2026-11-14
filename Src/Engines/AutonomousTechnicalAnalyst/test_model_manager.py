
from .model_manager import discover_free_chat_models


def main():
    print("=" * 60)
    print("OPENROUTER FREE MODEL DISCOVERY TEST")
    print("=" * 60)

    result = discover_free_chat_models(limit=10)

    print(f"\nStatus: {result['status']}")

    if result["status"] != "success":
        print(f"Error: {result['error']}")
        print("\nRESULT: MODEL DISCOVERY FAILED")
        return 1

    models = result["models"]

    print(f"\nDiscovered models: {len(models)}")

    for index, model in enumerate(models, start=1):
        print("-" * 50)
        print(f"Rank: {index}")
        print(f"Model ID: {model['id']}")
        print(f"Name: {model['name']}")
        print(f"Context length: {model['context_length']}")
        print(f"Pricing: {model['pricing']}")

    if not models:
        print("\nWARNING: No free text chat models found.")
        print("RESULT: DISCOVERY COMPLETED, NO MODELS FOUND")
        return 1

    print("\nRESULT: MODEL DISCOVERY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
