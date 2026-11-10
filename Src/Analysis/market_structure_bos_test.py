
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
