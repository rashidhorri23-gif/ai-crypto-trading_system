
def run_historical_confluence_audit(candles):
    print("\n" + "=" * 72)
    print("HISTORICAL 4H CONFLUENCE AUDIT")
    print("=" * 72)
    print("Method: Historical prefix scan")
    print("Timeframe: 4H")
    print("Directions: LONG and SHORT")
    print("OB/FVG overlap filter: STRICT")
    print("Live Trading: DISABLED")
    print("Archive Modification: NO")

    if not isinstance(candles, (list, tuple)) or len(candles) < 100:
        print("AUDIT STATUS: INSUFFICIENT_DATA")
        print("Minimum required candles: 100")
        print("Received candles:", len(candles) if candles else 0)
        return

    discovered_setups = {}
    error_counts = {}
    error_examples = []
    skipped_results = {
        "INVALID_RESULT": 0,
        "MISSING_SETUP": 0,
        "MISSING_SETUP_ID": 0,
        "MISSING_AVAILABLE_TIMESTAMP": 0,
        "DIRECTION_MISMATCH": 0,
    }

    total_evaluations = 0

    for end_index in range(100, len(candles) + 1):
        historical_candles = list(candles[:end_index])

        for direction in ("LONG", "SHORT"):
            total_evaluations += 1

            try:
                result = generate_four_hour_setup(
                    historical_candles,
                    direction,
                    setup_expiry_candles=24,
                    ob_lookback=10,
                    fvg_lookback=10,
                    risk_reward=2.0,
                )

                if not isinstance(result, dict):
                    skipped_results["INVALID_RESULT"] += 1
                    continue

                # Support both nested and flat result structures.
                nested_setup = result.get("setup")

                if isinstance(nested_setup, dict):
                    setup = dict(nested_setup)
                elif result.get("setup_id"):
                    setup = dict(result)
                else:
                    skipped_results["MISSING_SETUP"] += 1
                    continue

                setup_id = setup.get("setup_id")

                if not setup_id:
                    skipped_results["MISSING_SETUP_ID"] += 1
                    continue

                available_timestamp = setup.get(
                    "available_timestamp"
                )

                if available_timestamp is None:
                    skipped_results["MISSING_AVAILABLE_TIMESTAMP"] += 1
                    continue

                available_timestamp = int(available_timestamp)

                setup_direction = setup.get(
                    "direction",
                    direction
                )

                if setup_direction != direction:
                    skipped_results["DIRECTION_MISMATCH"] += 1
                    continue

                status = result.get(
                    "status",
                    setup.get("status", "UNKNOWN")
                )

                reason = result.get(
                    "reason",
                    setup.get("reason", "")
                )

                previous = discovered_setups.get(setup_id)

                if previous:
                    first_seen_index = previous["first_seen_index"]
                    observations = previous["observations"] + 1
                else:
                    first_seen_index = end_index - 1
                    observations = 1

                setup["status"] = status
                setup["reason"] = reason
                setup["first_seen_index"] = first_seen_index
                setup["last_seen_index"] = end_index - 1
                setup["observations"] = observations

                discovered_setups[setup_id] = setup

            except Exception as error:
                error_name = type(error).__name__

                error_counts[error_name] = (
                    error_counts.get(error_name, 0) + 1
                )

                if len(error_examples) < 5:
                    error_examples.append({
                        "index": end_index - 1,
                        "direction": direction,
                        "type": error_name,
                        "message": str(error),
                    })

    setups = list(discovered_setups.values())

    status_counts = {}
    direction_counts = {
        "LONG": 0,
        "SHORT": 0,
    }

    confluence_counts = {
        "CONFIRMED_CONFLUENCE": 0,
        "WATCH_NO_ZONE_OVERLAP": 0,
        "OTHER": 0,
    }

    for setup in setups:
        status = str(setup.get("status", "UNKNOWN"))
        direction = str(setup.get("direction", "UNKNOWN"))

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        if direction in direction_counts:
            direction_counts[direction] += 1

        if status in confluence_counts:
            confluence_counts[status] += 1
        else:
            confluence_counts["OTHER"] += 1

    print("\nAUDIT SUMMARY")
    print("-" * 72)
    print("4H Candles:", len(candles))
    print("Total Evaluations:", total_evaluations)
    print("Unique Setup IDs:", len(setups))

    print("\nSKIPPED RESULTS")
    print("-" * 72)

    for name, count in skipped_results.items():
        print(f"{name}: {count}")

    print("\nSETUPS BY DIRECTION")
    print("-" * 72)
    print("LONG:", direction_counts["LONG"])
    print("SHORT:", direction_counts["SHORT"])

    print("\nCONFLUENCE RESULTS")
    print("-" * 72)
    print(
        "Confirmed Confluence:",
        confluence_counts["CONFIRMED_CONFLUENCE"]
    )
    print(
        "No OB/FVG Overlap:",
        confluence_counts["WATCH_NO_ZONE_OVERLAP"]
    )
    print("Other Statuses:", confluence_counts["OTHER"])

    print("\nSTATUS DISTRIBUTION")
    print("-" * 72)

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    print("\nSETUP DETAILS")
    print("-" * 72)

    for setup in sorted(
        setups,
        key=lambda item: (
            int(item.get("available_timestamp", 0)),
            str(item.get("setup_id", ""))
        )
    ):
        print(
            "ID:", setup.get("setup_id"),
            "| Direction:", setup.get("direction"),
            "| Status:", setup.get("status"),
            "| Event:", setup.get("structure_event"),
            "| Available:", setup.get("available_timestamp"),
            "| Expiry:", setup.get("expiry_timestamp"),
            "| Observations:", setup.get("observations"),
        )

    if error_counts:
        print("\nAUDIT ERRORS")
        print("-" * 72)

        for error_name, count in sorted(error_counts.items()):
            print(f"{error_name}: {count}")

        print("\nERROR EXAMPLES")
        print("-" * 72)

        for example in error_examples:
            print(
                "Index:", example["index"],
                "| Direction:", example["direction"],
                "| Type:", example["type"],
                "| Message:", example["message"],
            )

    print("\nAUDIT COMPLETED")
    print("Live Trading: DISABLED")
    print("Archive Modification: NO")
    print("=" * 72)
