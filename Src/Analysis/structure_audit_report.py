def audit_failed_breaks(events, candles):
    results = []

    valid_events = (
        "BOS_BULLISH",
        "CHOCH_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BEARISH",
    )

    for event in events:
        event_type = event.get("event")

        if event_type not in valid_events:
            continue

        try:
            level = float(event["level"])
            event_index = int(event["index"])
        except (KeyError, TypeError, ValueError):
            continue

        if event_index < 0 or event_index >= len(candles):
            continue

        future_candles = candles[
            event_index + 1:
            event_index + 1 + FAILED_BREAK_LOOKAHEAD
        ]

        status = None
        return_index = None
        return_close = None

        for offset, candle in enumerate(future_candles, start=1):
            try:
                high = float(candle["high"])
                low = float(candle["low"])
                close = float(candle["close"])
            except (KeyError, TypeError, ValueError):
                continue

            if event_type.endswith("BULLISH"):
                if close <= level:
                    status = "BREAKOUT_FAILED"
                    return_index = event_index + offset
                    return_close = close
                    break

                if low <= level:
                    status = "RETEST_CONFIRMED"
                    return_index = event_index + offset
                    return_close = close
                    break

            else:
                if close >= level:
                    status = "BREAKOUT_FAILED"
                    return_index = event_index + offset
                    return_close = close
                    break

                if high >= level:
                    status = "RETEST_CONFIRMED"
                    return_index = event_index + offset
                    return_close = close
                    break

        if status is None:
            if len(future_candles) < FAILED_BREAK_LOOKAHEAD:
                status = "INSUFFICIENT_FUTURE_DATA"
            else:
                status = "NO_RETEST"

        results.append({
            "event": event_type,
            "level": level,
            "status": status,
            "return_timestamp": (
                candles[return_index]["timestamp"]
                if return_index is not None
                else None
            ),
            "return_close": return_close,
        })

    return results
def add_structure_report(
    lines,
    timeframe,
    structure_name,
    result,
    candles,
    candle_duration_seconds
):
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"{timeframe} {structure_name} STRUCTURE")
    lines.append("=" * 70)

    lines.append(f"Trend: {result['trend']}")
    lines.append(f"Event Bias: {result['event_bias']}")
    lines.append(f"Swing High Count: {result['high_count']}")
    lines.append(f"Swing Low Count: {result['low_count']}")

    lines.append("")
    lines.append("CONFIRMED SWINGS")
    lines.append("-" * 70)

    for swing in result["swings"]:
        confirmation_time = int(
            swing["confirmation_timestamp"]
        )

        available_time = int(
            swing.get(
                "available_timestamp",
                confirmation_time + candle_duration_seconds
            )
        )

        lines.append(
            f"{swing['type']} | "
            f"Price: {format_price(swing['price'])} | "
            f"Pivot: {format_time(swing['timestamp'])} | "
            f"Confirmed: {format_time(confirmation_time)} | "
            f"Available: {format_time(available_time)} | "
            f"Status: {swing.get('status', 'UNKNOWN')} | "
            f"Index: {swing['index']}"
        )

    lines.append("")
    lines.append("STRUCTURE EVENTS")
    lines.append("-" * 70)

    for event in result["events"]:
        event_name = event["event"]

        if event_name == "AMBIGUOUS_DOUBLE_BREAK":
            bullish_levels = event.get(
                "bullish_levels",
                []
            )

            bearish_levels = event.get(
                "bearish_levels",
                []
            )

            if not bullish_levels and event.get("bullish_level") is not None:
                bullish_levels = [event["bullish_level"]]

            if not bearish_levels and event.get("bearish_level") is not None:
                bearish_levels = [event["bearish_level"]]

            bullish_text = ", ".join(
                format_price(level)
                for level in bullish_levels
            ) or "N/A"

            bearish_text = ", ".join(
                format_price(level)
                for level in bearish_levels
            ) or "N/A"

            lines.append(
                f"{event_name} | "
                f"Time: {format_time(event['timestamp'])} | "
                f"Bullish Levels: [{bullish_text}] | "
                f"Bearish Levels: [{bearish_text}] | "
                f"Close: {format_price(event.get('close'))}"
            )

            continue

        lines.append(
            f"{event_name} | "
            f"Level: {format_price(event['level'])} | "
            f"Close: {format_price(event['close'])} | "
            f"Time: {format_time(event['timestamp'])} | "
            f"Available: {format_time(event['available_timestamp'])}"
        )

    lines.append("")
    lines.append("BREAK RETEST AUDIT")
    lines.append("-" * 70)

    failed_results = audit_failed_breaks(
        result["events"],
        candles
    )

    for item in failed_results:
        line = (
            f"{item['event']} | "
            f"Level: {format_price(item['level'])} | "
            f"Status: {item['status']}"
        )

        if item["return_timestamp"] is not None:
            line += (
                f" | Return Time: "
                f"{format_time(item['return_timestamp'])}"
                f" | Return Close: "
                f"{format_price(item['return_close'])}"
            )

        lines.append(line)

    if not failed_results:
        lines.append("No directional break events found.")
