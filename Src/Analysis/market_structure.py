def create_analysis(data, timeframe=None):
    validated = validate_input_data(data)

    timestamp_is_milliseconds = (
        int(validated["timestamp"].median())
        >= 100_000_000_000
    )

    if timeframe is None:
        timeframe = infer_timeframe(validated)
    else:
        timeframe = validate_timeframe(timeframe)

    engine_data = validated.copy()

    if timestamp_is_milliseconds:
        if (engine_data["timestamp"] % 1000 != 0).any():
            raise ValueError(
                "Millisecond timestamps must align with whole seconds."
            )

        engine_data["timestamp"] = (
            engine_data["timestamp"] // 1000
        )

    candles = prepare_candles(engine_data)

    analysis = analyze_daily_structure(
        candles,
        timeframe=timeframe
    )

    if analysis.get("status") != "ANALYZED":
        raise RuntimeError(
            "Daily structure analysis did not complete successfully."
        )

    if timestamp_is_milliseconds:
        analysis = scale_analysis_timestamps(
            analysis,
            1000
        )

    return validated, analysis
