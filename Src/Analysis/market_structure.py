def detect_market_structure(
    data,
    left_bars=DEFAULT_LEFT_BARS,
    right_bars=DEFAULT_RIGHT_BARS,
    timeframe=None
):
    if isinstance(left_bars, bool) or not isinstance(left_bars, int):
        raise ValueError("left_bars must be an integer.")

    if isinstance(right_bars, bool) or not isinstance(right_bars, int):
        raise ValueError("right_bars must be an integer.")

    if left_bars < 1 or right_bars < 1:
        raise ValueError("Swing periods must be positive.")

    result, analysis = create_analysis(
        data,
        timeframe=timeframe,
        left_bars=left_bars,
        right_bars=right_bars,
    )

    result = add_swing_columns(
        result,
        analysis
    )

    result = add_event_columns(
        result,
        analysis
    )

    return result
