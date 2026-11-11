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
        retest_wick = None

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
                    retest_wick = low
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
                    retest_wick = high
                    break

        if status is None:
            if len(future_candles) < FAILED_BREAK_LOOKAHEAD:
                status = "INSUFFICIENT_FUTURE_DATA"
            else:
                status = "NO_RETEST"

        results.append({
            "event": event_type,
            "level": level,
            "timestamp": event["timestamp"],
            "status": status,
            "return_timestamp": (
                candles[return_index]["timestamp"]
                if return_index is not None
                else None
            ),
            "return_close": return_close,
            "retest_wick": retest_wick,
        })

    return results
