
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.Analysis.four_hour_setup import generate_four_hour_setup


SYMBOL = "BTC-USDT"
TIMEFRAME = "4H"

ARCHIVE_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Historical"
    / "BTC_USDT_SPOT_4H.csv"
)

MINIMUM_CANDLES = 100


def load_four_hour_candles():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"4H archive not found: {ARCHIVE_PATH}"
        )

    candles = []

    with ARCHIVE_PATH.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                candle = {
                    "timestamp": int(
                        float(row["timestamp"])
                    ),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }

                candles.append(candle)

            except (KeyError, TypeError, ValueError):
                continue

    candles.sort(
        key=lambda item: item["timestamp"]
    )

    unique_candles = []
    seen_timestamps = set()

    for candle in candles:
        timestamp = candle["timestamp"]

        if timestamp in seen_timestamps:
            continue

        seen_timestamps.add(timestamp)
        unique_candles.append(candle)

    return unique_candles


def run_historical_confluence_audit(candles):
    print("\n" + "=" * 72, flush=True)
    print("HISTORICAL 4H CONFLUENCE AUDIT", flush=True)
    print("=" * 72, flush=True)
    print("Symbol:", SYMBOL, flush=True)
    print("Timeframe:", TIMEFRAME, flush=True)
    print("Method: Historical prefix scan", flush=True)
    print("Directions: LONG and SHORT", flush=True)
    print("OB/FVG overlap filter: STRICT", flush=True)
    print("Lookahead: DISABLED", flush=True)
    print("Live Trading: DISABLED", flush=True)
    print("Archive Modification: NO", flush=True)

    if not isinstance(candles, (list, tuple)):
        print("AUDIT STATUS: INVALID_DATA", flush=True)
        return

    if len(candles) < MINIMUM_CANDLES:
        print("AUDIT STATUS: INSUFFICIENT_DATA", flush=True)
        print("Minimum required:", MINIMUM_CANDLES, flush=True)
        print("Received:", len(candles), flush=True)
        return

    total_prefixes = len(candles) - MINIMUM_CANDLES + 1
    total_evaluations = 0

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

    print("\nAUDIT STARTED", flush=True)
    print("Historical Candles:", len(candles), flush=True)
    print("Evaluated Prefixes:", total_prefixes, flush=True)
    print("Expected Evaluations:", total_prefixes * 2, flush=True)

    for end_index in range(MINIMUM_CANDLES, len(candles) + 1):
        historical_candles = candles[:end_index]

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

                setup_direction = setup.get("direction", direction)

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

                if previous is None:
                    setup["first_seen_index"] = end_index - 1
                    setup["observations"] = 1
                else:
                    setup["first_seen_index"] = (
                        previous["first_seen_index"]
                    )
                    setup["observations"] = (
                        previous["observations"] + 1
                    )

                setup["available_timestamp"] = available_timestamp
                setup["direction"] = setup_direction
                setup["status"] = status
                setup["reason"] = reason
                setup["last_seen_index"] = end_index - 1

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

        completed_prefixes = end_index - MINIMUM_CANDLES + 1

        if (
            completed_prefixes % 25 == 0
            or completed_prefixes == total_prefixes
        ):
            print(
                "Progress:",
                completed_prefixes,
                "/",
                total_prefixes,
                "| Evaluations:",
                total_evaluations,
                "| Setups:",
                len(discovered_setups),
                "| Errors:",
                sum(error_counts.values()),
                flush=True,
            )

    setups = list(discovered_setups.values())

    direction_counts = {
        "LONG": 0,
        "SHORT": 0,
    }

    status_counts = {}

    for setup in setups:
        direction = str(setup.get("direction", "UNKNOWN"))
        status = str(setup.get("status", "UNKNOWN"))

        if direction in direction_counts:
            direction_counts[direction] += 1

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    confirmed = status_counts.get(
        "CONFIRMED_CONFLUENCE", 0
    )

    no_overlap = status_counts.get(
        "WATCH_NO_ZONE_OVERLAP", 0
    )

    other = len(setups) - confirmed - no_overlap

    print("\nHISTORICAL SUMMARY", flush=True)
    print("-" * 72, flush=True)
    print("Historical Candles:", len(candles), flush=True)
    print("Evaluated Candle Prefixes:", total_prefixes, flush=True)
    print("Total Evaluations:", total_evaluations, flush=True)
    print("Unique Setups Discovered:", len(setups), flush=True)
    print("Engine Errors:", sum(error_counts.values()), flush=True)

    print("\nSETUPS BY DIRECTION", flush=True)
    print("-" * 72, flush=True)
    print("LONG:", direction_counts["LONG"], flush=True)
    print("SHORT:", direction_counts["SHORT"], flush=True)

    print("\nCONFLUENCE RESULTS", flush=True)
    print("-" * 72, flush=True)
    print("Confirmed Confluence:", confirmed, flush=True)
    print("Watch Without Zone Overlap:", no_overlap, flush=True)
    print("Other Statuses:", other, flush=True)

    print("\nSTATUS DISTRIBUTION", flush=True)
    print("-" * 72, flush=True)

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}", flush=True)

    print("\nSKIPPED RESULTS", flush=True)
    print("-" * 72, flush=True)

    for name, count in skipped_results.items():
        print(f"{name}: {count}", flush=True)

    print("\nSETUP DETAILS", flush=True)
    print("-" * 72, flush=True)

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
            flush=True
        )

    if error_counts:
        print("\nAUDIT ERRORS", flush=True)

        for name, count in sorted(error_counts.items()):
            print(f"{name}: {count}", flush=True)

        print("\nERROR EXAMPLES", flush=True)

        for example in error_examples:
            print(
                "Index:", example["index"],
                "| Direction:", example["direction"],
                "| Type:", example["type"],
                "| Message:", example["message"],
                flush=True
            )

    print("\nAUDIT COMPLETED", flush=True)
    print("Live Trading: DISABLED", flush=True)
    print("Archive Modification: NO", flush=True)
    print("=" * 72, flush=True)


def main():
    print("Starting historical confluence audit", flush=True)

    candles = load_four_hour_candles()

    print("Loaded 4H candles:", len(candles), flush=True)

    run_historical_confluence_audit(candles)


if __name__ == "__main__":
    main()
