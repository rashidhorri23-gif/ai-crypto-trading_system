def analyze_daily_structure(
    candles,
    timeframe="1D",
    internal_left_bars=2,
    internal_right_bars=2,
):
    candles = validate_candles(candles)

    for name, value in (
        ("internal_left_bars", internal_left_bars),
        ("internal_right_bars", internal_right_bars),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    candle_duration = infer_candle_duration(candles)

    internal_swings = detect_swings(
        candles,
        left_bars=internal_left_bars,
        right_bars=internal_right_bars,
    )

    external_swings = detect_swings(
        candles,
        left_bars=5,
        right_bars=5,
    )

    internal_structure = classify_market_structure(internal_swings)
    external_structure = classify_market_structure(external_swings)

    internal_events = analyze_structure_events(
        candles,
        internal_swings,
        candle_duration,
    )

    external_events = analyze_structure_events(
        candles,
        external_swings,
        candle_duration,
    )

    return {
        "timeframe": timeframe,
        "internal": {
            "swings": internal_swings,
            "structure": internal_structure,
            "events": internal_events,
        },
        "external": {
            "swings": external_swings,
            "structure": external_structure,
            "events": external_events,
        },
        "context": {
            "internal_trend": internal_structure.get("trend"),
            "external_trend": external_structure.get("trend"),
        },
        "trend": external_structure.get("trend"),
        "allowed_direction": "BOTH",
        "trade_directions_to_evaluate": ["LONG", "SHORT"],
        "status": "ANALYZED",
    }
