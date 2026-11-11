
FAILED_BREAK_LOOKAHEAD = 3


def audit_failed_breaks(events, candles):
    results = []

    valid_events = {
        "BOS_BULLISH",
        "CHOCH_BULLISH",
        "BOS_BEARISH",
        "CHOCH_BEARISH",
    }

    if isinstance(candles, pd.DataFrame):
        candle_records = candles.to_dict("records")
    else:
        candle_records = list(candles)

    if FAILED_BREAK_LOOKAHEAD < 1:
        raise ValueError(
            "FAILED_BREAK_LOOKAHEAD must be positive."
        )

    required_columns = {
        "timestamp",
        "high",
        "low",
        "close",
    }

    for event in events:
        event_type = event.get("event")

        if event_type not in valid_events:
            continue

        try:
            level = float(event["level"])
            event_index = int(event["index"])
        except (KeyError, TypeError, ValueError):
            continue

        if not np.isfinite(level) or level <= 0:
            continue

        if event_index < 0 or event_index >= len(candle_records):
            continue

        missing_columns = (
            required_columns
            - set(candle_records[event_index].keys())
        )

        if missing_columns:
            raise ValueError(
                f"Missing candle fields: {missing_columns}"
            )

        event_candle = candle_records[event_index]

        if event.get("timestamp") is not None:
            if int(event["timestamp"]) != int(
                event_candle["timestamp"]
            ):
                raise ValueError(
                    "Event timestamp does not match its candle index."
                )

        future_start = event_index + 1

        future_end = min(
            future_start + FAILED_BREAK_LOOKAHEAD,
            len(candle_records)
        )

        future_candles = candle_records[
            future_start:future_end
        ]

        status = None
        return_index = None
        return_close = None
        return_timestamp = None

        for offset, candle in enumerate(
            future_candles,
            start=1
        ):
            missing_columns = (
                required_columns - set(candle.keys())
            )

            if missing_columns:
                raise ValueError(
                    f"Missing future candle fields: {missing_columns}"
                )

            try:
                high = float(candle["high"])
                low = float(candle["low"])
                close = float(candle["close"])
                timestamp = int(candle["timestamp"])

            except (TypeError, ValueError):
                raise ValueError(
                    "Invalid future candle data during break audit."
                )

            if not all(
                np.isfinite(value)
                for value in (high, low, close)
            ):
                raise ValueError(
                    "Non-finite future candle prices found."
                )

            if high < low:
                raise ValueError(
                    "Invalid future candle: high is below low."
                )

            if event_type.endswith("BULLISH"):

                if close <= level:
                    status = "BREAKOUT_FAILED"

                elif low <= level:
                    status = "RETEST_CONFIRMED"

            else:

                if close >= level:
                    status = "BREAKOUT_FAILED"

                elif high >= level:
                    status = "RETEST_CONFIRMED"

            if status is not None:
                return_index = event_index + offset
                return_close = close
                return_timestamp = timestamp
                break

        if status is None:

            if len(future_candles) < FAILED_BREAK_LOOKAHEAD:
                status = "INSUFFICIENT_FUTURE_DATA"

            else:
                status = (
                    "HELD_BEYOND_LEVEL_FOR_3_CANDLES"
                )

        results.append({
            "event": event_type,
            "level": level,
            "event_index": event_index,
            "event_timestamp": int(
                event_candle["timestamp"]
            ),
            "status": status,
            "candles_checked": len(future_candles),
            "bars_to_return": (
                return_index - event_index
                if return_index is not None
                else None
            ),
            "return_index": return_index,
            "return_timestamp": return_timestamp,
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

            if (
                not bullish_levels
                and event.get("bullish_level") is not None
            ):
                bullish_levels = [
                    event["bullish_level"]
                ]

            if (
                not bearish_levels
                and event.get("bearish_level") is not None
            ):
                bearish_levels = [
                    event["bearish_level"]
                ]

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

        event_timestamp = event["timestamp"]

        available_timestamp = event.get(
            "available_timestamp"
        )

        if available_timestamp is None:
            available_timestamp = (
                int(event_timestamp)
                + candle_duration_seconds
            )

        lines.append(
            f"{event_name} | "
            f"Level: {format_price(event['level'])} | "
            f"Close: {format_price(event['close'])} | "
            f"Time: {format_time(event_timestamp)} | "
            f"Available: {format_time(available_timestamp)}"
        )

    lines.append("")
    lines.append("BREAK RETEST AUDIT")
    lines.append("-" * 70)

    failed_results = audit_failed_breaks(
        result["events"],
        candles
    )

    status_counts = {}

    for item in failed_results:

        status = item["status"]

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        line = (
            f"{item['event']} | "
            f"Level: {format_price(item['level'])} | "
            f"Status: {status} | "
            f"Checked: {item['candles_checked']} candles"
        )

        if item["return_timestamp"] is not None:

            line += (
                f" | Return Time: "
                f"{format_time(item['return_timestamp'])}"
                f" | Return Close: "
                f"{format_price(item['return_close'])}"
                f" | Bars: {item['bars_to_return']}"
            )

        lines.append(line)

    lines.append("")
    lines.append("AUDIT SUMMARY")
    lines.append("-" * 70)

    lines.append(
        f"Total Audited Events: {len(failed_results)}"
    )

    for status, count in sorted(status_counts.items()):
        lines.append(
            f"{status}: {count}"
        )

    if not failed_results:
        lines.append(
            "No directional break events found."
        )
