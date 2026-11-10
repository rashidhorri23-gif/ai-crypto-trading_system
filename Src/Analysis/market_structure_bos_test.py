
def test_single_break_per_level():
    data = build_data()

    data.at[3, "high"] = 110.0

    for index in range(7, 12):
        data.at[index, "open"] = 110.5
        data.at[index, "close"] = 111.0
        data.at[index, "high"] = 111.5
        data.at[index, "low"] = 109.5

    result = run_structure(data)

    events = result[
        result["structure_event"] != "NONE"
    ]

    check(
        len(events) == 1,
        "The same broken level generated multiple events."
    )

    check(
        events.iloc[0]["structure_event"] == "BOS_BULLISH",
        "Unexpected structure event."
    )

    print("SINGLE BREAK PER LEVEL: PASS")

def main():
    print()
    print("BOS AND CHOCH UNIT TEST")
    print("=======================")

    test_functions = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    passed = 0
    failed = 0

    for name, function in test_functions:
        try:
            function()
            passed += 1
        except Exception as error:
            failed += 1
            print(f"{name}: FAIL - {error}")

    print()
    print("FINAL BOS AND CHOCH REPORT")
    print("==========================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0 and passed == 8:
        print("OVERALL BOS AND CHOCH STATUS: PASS")
    else:
        print("OVERALL BOS AND CHOCH STATUS: FAIL")
        raise SystemExit(1)

    print("ARCHIVE MODIFICATION: NO")


if __name__ == "__main__":
    main()
