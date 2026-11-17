
def fetch_candles_since(bar, initial_candles, last_processed, interval):
    if not initial_candles:
        raise RuntimeError(
            "No initial candles available for history recovery."
        )

    candles_by_timestamp = {
        candle["timestamp"]: candle
        for candle in initial_candles
    }

    if len(candles_by_timestamp) != len(initial_candles):
        log("Duplicate candle timestamps detected and removed.")

    if last_processed is None:
        return sorted(
            candles_by_timestamp.values(),
            key=lambda item: item["timestamp"],
        )

    if isinstance(last_processed, bool) or not isinstance(
        last_processed, int
    ):
        raise RuntimeError("Invalid last processed timestamp.")

    def find_gaps(candles):
        ordered = sorted(
            candles,
            key=lambda item: item["timestamp"],
        )

        gaps = []

        for index in range(1, len(ordered)):
            previous_timestamp = ordered[index - 1]["timestamp"]
            current_timestamp = ordered[index]["timestamp"]

            difference = current_timestamp - previous_timestamp

            if difference != interval:
                if difference < interval or difference % interval != 0:
                    raise RuntimeError(
                        "Invalid candle timestamp sequence detected."
                    )

                missing_count = difference // interval - 1

                gaps.append({
                    "start": previous_timestamp,
                    "end": current_timestamp,
                    "missing": missing_count,
                })

        return ordered, gaps

    pages = 0

    while True:
        ordered, gaps = find_gaps(
            list(candles_by_timestamp.values())
        )

        oldest_timestamp = ordered[0]["timestamp"]
        latest_timestamp = ordered[-1]["timestamp"]

        required_oldest = last_processed

        if oldest_timestamp > required_oldest:
            if pages >= MAX_HISTORY_PAGES:
                raise RuntimeError(
                    "History recovery exceeded the page limit. "
                    "Checkpoint will not advance."
                )

            log(
                "Historical coverage gap detected. "
                "Recovering older candles from OKX."
            )

            older_candles = request_candle_page(
                bar=bar,
                limit=300,
                after=oldest_timestamp * 1000,
                history=True,
            )

            older_candles = [
                candle
                for candle in older_candles
                if candle["timestamp"] < oldest_timestamp
            ]

            if not older_candles:
                raise RuntimeError(
                    "OKX returned no older candles during recovery. "
                    "Checkpoint will not advance."
                )

            previous_oldest = oldest_timestamp

            for candle in older_candles:
                candles_by_timestamp[candle["timestamp"]] = candle

            new_oldest = min(candles_by_timestamp)

            if new_oldest >= previous_oldest:
                raise RuntimeError(
                    "History pagination made no progress."
                )

            pages += 1

            log(
                f"History page {pages} recovered. "
                f"Oldest candle: {new_oldest}"
            )

            continue

        if not gaps:
            break

        gap = gaps[0]

        log(
            f"Internal candle gap detected. "
            f"Missing candles: {gap['missing']}"
        )

        if pages >= MAX_HISTORY_PAGES:
            raise RuntimeError(
                "History recovery exceeded the page limit "
                "while repairing an internal gap."
            )

        recovered_page = request_candle_page(
            bar=bar,
            limit=300,
            after=gap["start"] * 1000,
            history=True,
        )

        recovered_page = [
            candle
            for candle in recovered_page
            if gap["start"] < candle["timestamp"] < gap["end"]
        ]

        if not recovered_page:
            raise RuntimeError(
                "OKX could not recover an internal candle gap. "
                "Checkpoint will not advance."
            )

        previous_count = len(candles_by_timestamp)

        for candle in recovered_page:
            candles_by_timestamp[candle["timestamp"]] = candle

        if len(candles_by_timestamp) == previous_count:
            raise RuntimeError(
                "Internal gap recovery made no progress."
            )

        pages += 1

    ordered, remaining_gaps = find_gaps(
        list(candles_by_timestamp.values())
    )

    if remaining_gaps:
        raise RuntimeError(
            "Candle history still contains unresolved gaps. "
            "Checkpoint will not advance."
        )

    if ordered[-1]["timestamp"] < last_processed:
        raise RuntimeError(
            "Latest available candle is older than the checkpoint."
        )

    log(
        f"History validation completed. "
        f"Total candles: {len(ordered)}. "
        f"Recovered pages: {pages}."
    )

    return ordered
