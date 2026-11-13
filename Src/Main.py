
import sys
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Src.Config import settings

from Src.Data.okx_client import (
    fetch_historical_candles,
    fetch_new_candles,
)

from Src.Data.archive import (
    save_candles,
    get_latest_timestamp,
)

from Src.Analysis.daily_structure import (
    analyze_daily_structure,
)

from Src.Analysis.market_direction import (
    determine_market_direction,
)

from Src.Analysis.four_hour_setup import (
    generate_four_hour_setup,
)

from Src.Analysis.entry_confirmation import (
    confirm_lower_timeframe_entry,
)

from Src.Analysis.setup_deduplication import (
    deduplicate_setups,
    print_deduplication_report,
)


MARKET = "SPOT"
SYMBOL = "BTC-USDT"

RUN_HISTORICAL_CONFLUENCE_AUDIT = True
RUN_4H_MODE_COMPARISON_AUDIT = True

TIMEFRAME_CONFIG = {
    "1D": {
        "target": 365,
        "interval_ms": 86400000,
        "filename": "BTC_USDT_SPOT_1D.csv",
    },
    "4H": {
        "target": 540,
        "interval_ms": 14400000,
        "filename": "BTC_USDT_SPOT_4H.csv",
    },
    "1H": {
        "target": 720,
        "interval_ms": 3600000,
        "filename": "BTC_USDT_SPOT_1H.csv",
    },
    "15m": {
        "target": 960,
        "interval_ms": 900000,
        "filename": "BTC_USDT_SPOT_15m.csv",
    },
    "5m": {
        "target": 1016,
        "interval_ms": 300000,
        "filename": "BTC_USDT_SPOT_5m.csv",
    },
}


def get_archive_path(filename):
    return PROJECT_ROOT / "Data" / "Archive" / filename


def count_archive_rows(filename):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        return 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        return sum(1 for row in reader)


def initialize_archive(timeframe, config):
    filename = config["filename"]
    target = config["target"]
    archive_path = get_archive_path(filename)

    if timeframe == "5m" and not archive_path.exists():
        raise RuntimeError(
            "Existing 5m archive is missing. "
            "Automatic recreation is disabled."
        )

    current_count = count_archive_rows(filename)

    if current_count >= target:
        print(
            f"{timeframe}: Existing archive contains "
            f"{current_count} candles. Initial seed not required."
        )
        return

    print(
        f"{timeframe}: Initializing historical archive. "
        f"Target: {target} candles."
    )

    candles = fetch_historical_candles(
        market=MARKET,
        timeframe=timeframe,
        total_candles=target,
        page_size=100,
    )

    if len(candles) != target:
        raise RuntimeError(
            f"{timeframe}: Historical seed incomplete. "
            f"Expected {target}, received {len(candles)}. "
            "Archive was not updated."
        )

    result = save_candles(candles, filename)

    print(
        f"{timeframe}: Historical seed saved. "
        f"New candles: {result['new_candles']}, "
        f"total candles: {result['total_candles']}"
    )


def update_archive(timeframe, config):
    filename = config["filename"]

    last_timestamp = get_latest_timestamp(filename)

    if last_timestamp is None:
        raise RuntimeError(
            f"{timeframe}: Archive has no latest timestamp."
        )

    last_time = datetime.fromtimestamp(
        last_timestamp / 1000,
        tz=timezone.utc,
    )

    print(f"\n{timeframe} INCREMENTAL UPDATE")
    print("============================")
    print(f"Last Archived UTC: {last_time.isoformat()}")

    candles = fetch_new_candles(
        market=MARKET,
        timeframe=timeframe,
        last_timestamp=last_timestamp,
        page_size=100,
    )

    print(f"New Candles Retrieved: {len(candles)}")

    if candles:
        result = save_candles(candles, filename)

        print(f"New Candles Saved: {result['new_candles']}")
        print(f"Total Archived Candles: {result['total_candles']}")

        newest_time = datetime.fromtimestamp(
            candles[-1]["timestamp"] / 1000,
            tz=timezone.utc,
        )

        print(f"Newest Candle UTC: {newest_time.isoformat()}")

    else:
        print("No new completed candles found.")
        print("Archive remains unchanged.")


def audit_archive(filename, interval_ms):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        raise RuntimeError(
            f"Archive file not found: {filename}"
        )

    timestamps = []
    invalid_rows = 0
    misaligned_timestamps = 0
    total_rows = 0

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            try:
                timestamp = int(row["timestamp"])

                open_price = float(row["open"])
                high_price = float(row["high"])
                low_price = float(row["low"])
                close_price = float(row["close"])
                volume = float(row["volume"])

                if (
                    min(
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                    ) <= 0
                    or volume < 0
                    or high_price < max(
                        open_price,
                        close_price,
                        low_price,
                    )
                    or low_price > min(
                        open_price,
                        close_price,
                        high_price,
                    )
                ):
                    invalid_rows += 1

                timestamps.append(timestamp)

                if timestamp % interval_ms != 0:
                    misaligned_timestamps += 1

            except (ValueError, KeyError, TypeError):
                invalid_rows += 1

    if not timestamps:
        raise RuntimeError(
            f"{filename}: Archive contains no valid timestamps."
        )

    unique_timestamps = sorted(set(timestamps))

    duplicate_count = (
        len(timestamps) - len(unique_timestamps)
    )

    missing_candles = 0
    gap_count = 0

    for index in range(1, len(unique_timestamps)):
        difference = (
            unique_timestamps[index]
            - unique_timestamps[index - 1]
        )

        if difference > interval_ms:
            missing = (
                difference // interval_ms
            ) - 1

            missing_candles += missing
            gap_count += 1

        elif difference < interval_ms:
            invalid_rows += 1

    first_time = datetime.fromtimestamp(
        unique_timestamps[0] / 1000,
        tz=timezone.utc,
    )

    last_time = datetime.fromtimestamp(
        unique_timestamps[-1] / 1000,
        tz=timezone.utc,
    )

    expected_count = (
        (
            unique_timestamps[-1]
            - unique_timestamps[0]
        )
        // interval_ms
    ) + 1

    continuous = (
        len(unique_timestamps) == expected_count
    )

    passed = (
        duplicate_count == 0
        and missing_candles == 0
        and invalid_rows == 0
        and misaligned_timestamps == 0
        and continuous
    )

    print("\nARCHIVE AUDIT REPORT")
    print("====================")
    print(f"Archive File: {filename}")
    print(f"Total Rows: {total_rows}")
    print(f"Unique Candles: {len(unique_timestamps)}")
    print(f"Duplicate Rows: {duplicate_count}")
    print(f"Missing Candles: {missing_candles}")
    print(f"Gap Segments: {gap_count}")
    print(f"Invalid Rows: {invalid_rows}")
    print(f"Misaligned Timestamps: {misaligned_timestamps}")
    print(f"First Candle UTC: {first_time.isoformat()}")
    print(f"Last Candle UTC: {last_time.isoformat()}")
    print(f"Expected Continuous Count: {expected_count}")
    print(f"Continuous Timeline: {continuous}")
    print(
        f"AUDIT STATUS: "
        f"{'PASS' if passed else 'ISSUES DETECTED'}"
    )

    return passed


def load_structure_candles(filename, limit):
    archive_path = get_archive_path(filename)

    if not archive_path.exists():
        raise RuntimeError(
            f"Structure archive not found: {filename}"
        )

    candles = []

    with archive_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                candle = {
                    "timestamp": int(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }

                candles.append(candle)

            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError(
                    f"Invalid candle in {filename}: {error}"
                )

    candles.sort(
        key=lambda candle: candle["timestamp"]
    )

    if len(candles) < limit:
        raise RuntimeError(
            f"{filename}: Insufficient candles. "
            f"Required {limit}, found {len(candles)}."
        )

    return candles[-limit:]


def run_market_direction_analysis():
    print("\nREAL MARKET DIRECTION ANALYSIS")
    print("==============================")

    daily_candles = load_structure_candles(
        TIMEFRAME_CONFIG["1D"]["filename"],
        90,
    )

    four_hour_candles = load_structure_candles(
        TIMEFRAME_CONFIG["4H"]["filename"],
        540,
    )

    print(f"Daily Candles: {len(daily_candles)}")
    print(f"4H Candles: {len(four_hour_candles)}")

    daily_result = analyze_daily_structure(
        daily_candles,
        timeframe="1D",
    )

    four_hour_result = analyze_daily_structure(
        four_hour_candles,
        timeframe="4H",
    )

    decision = determine_market_direction(
        daily_result,
        four_hour_result,
    )

    print("\nDAILY MARKET STRUCTURE")
    print("======================")
    print("Trend:", decision["daily_bias"])
    print("Event Bias:", decision["daily_event_bias"])
    print("Reason:", decision["daily_structure_reason"])

    print("\n4H MARKET STRUCTURE")
    print("==================")
    print("Trend:", decision["four_hour_bias"])
    print("Event Bias:", decision["four_hour_event_bias"])
    print("Reason:", decision["four_hour_structure_reason"])

    print("\nFINAL MARKET DIRECTION")
    print("======================")
    print("Allowed Direction:", decision["allowed_direction"])
    print("Status:", decision["status"])
    print("Reason:", decision["reason"])
    print(
        "Trade Directions:",
        decision["trade_directions_to_evaluate"],
    )

    return decision, four_hour_candles


def run_four_hour_mode_comparison_audit(candles):
    print("\n4H MITIGATION MODE COMPARISON")
    print("=============================")

    print("Symbol:", SYMBOL)
    print("Timeframe: 4H")
    print("Historical Candles:", len(candles))
    print("Modes: A, B, C")
    print("Directions: LONG, SHORT")
    print("Same Candle Dataset: YES")
    print("Live Trading: DISABLED")
    print("Entry Execution: DISABLED")

    modes = ("A", "B", "C")
    directions = ("LONG", "SHORT")

    comparison_results = {}

    for mode in modes:
        print("\n")
        print("#" * 60)
        print(f"MITIGATION MODE: {mode}")
        print("#" * 60)

        for direction in directions:
            key = f"{mode}_{direction}"

            print("\n----------------------------------------")
            print(f"MODE: {mode} | DIRECTION: {direction}")
            print("----------------------------------------")

            try:
                result = generate_four_hour_setup(
                    candles=candles,
                    allowed_direction=direction,
                    setup_expiry_candles=24,
                    ob_lookback=10,
                    fvg_lookback=10,
                    risk_reward=2.0,
                    symbol=SYMBOL,
                    mitigation_mode=mode,
                )

                setup = result.get("setup")
                diagnostics = result.get("diagnostics", {})

                comparison_results[key] = {
                    "mode": mode,
                    "direction": direction,
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "setup": setup,
                    "diagnostics": diagnostics,
                }

                print("Status:", result.get("status"))
                print("Reason:", result.get("reason"))
                print(
                    "Setup Generated:",
                    isinstance(setup, dict),
                )

                if isinstance(setup, dict):
                    print("Setup ID:", setup.get("setup_id"))
                    print(
                        "Structure Event:",
                        setup.get("structure_event"),
                    )
                    print(
                        "Structure Event Timestamp:",
                        setup.get("structure_event_timestamp"),
                    )
                    print("Entry Zone:", setup.get("entry_zone"))
                    print("Order Block:", setup.get("order_block"))
                    print("FVG:", setup.get("fvg"))
                    print("Zone Overlap:", setup.get("zone_overlap"))
                    print("Stop Loss:", setup.get("stop_loss"))
                    print("Take Profit:", setup.get("take_profit"))

                print("\nDIAGNOSTICS")
                print("-----------")
                print(
                    json.dumps(
                        diagnostics,
                        indent=2,
                        ensure_ascii=True,
                        default=str,
                    )
                )

            except Exception as error:
                comparison_results[key] = {
                    "mode": mode,
                    "direction": direction,
                    "error": str(error),
                }

                print("ENGINE ERROR:", str(error))

    print("\n4H MODE COMPARISON SUMMARY")
    print("==========================")

    for mode in modes:
        for direction in directions:
            key = f"{mode}_{direction}"
            record = comparison_results.get(key, {})

            if "error" in record:
                print(
                    f"{key}: ERROR - {record['error']}"
                )
                continue

            setup = record.get("setup")

            print(
                f"{key}: "
                f"Status={record.get('status')} | "
                f"Setup={'YES' if isinstance(setup, dict) else 'NO'}"
            )

    print("\n4H MODE COMPARISON COMPLETED")
    print("============================")
    print("No trades were executed.")
    print("Live Trading: DISABLED")

    return comparison_results


def run_historical_confluence_audit(candles):
    print("\nHISTORICAL 4H CONFLUENCE AUDIT")
    print("================================")

    print("Symbol:", SYMBOL)
    print("Timeframe: 4H")
    print("Historical Candles:", len(candles))
    print("Method: Candle-by-Candle Historical Evaluation")
    print("Lookahead: DISABLED")

    if len(candles) < 100:
        raise RuntimeError(
            "Insufficient candles for historical confluence audit."
        )

    discovered_setups = {}
    engine_errors = []
    evaluated_prefixes = 0

    for end_index in range(100, len(candles) + 1):
        historical_candles = candles[:end_index]
        current_timestamp = historical_candles[-1]["timestamp"]

        evaluated_prefixes += 1

        for direction in ("LONG", "SHORT"):
            try:
                result = generate_four_hour_setup(
                    candles=historical_candles,
                    allowed_direction=direction,
                    setup_expiry_candles=24,
                    ob_lookback=10,
                    fvg_lookback=10,
                    risk_reward=2.0,
                    symbol=SYMBOL,
                    mitigation_mode="A",
                )

                setup = result.get("setup")

                if not isinstance(setup, dict):
                    continue

                setup_id = setup.get("setup_id")

                if not setup_id:
                    continue

                available_timestamp = setup.get(
                    "available_timestamp"
                )

                if available_timestamp is None:
                    continue

                if available_timestamp > current_timestamp:
                    continue

                if setup_id in discovered_setups:
                    continue

                status = result.get(
                    "status",
                    setup.get("status", "UNKNOWN"),
                )

                setup_record = dict(setup)
                setup_record["setup_id"] = setup_id
                setup_record["symbol"] = setup_record.get(
                    "symbol",
                    SYMBOL,
                )
                setup_record["timeframe"] = setup_record.get(
                    "timeframe",
                    "4H",
                )
                setup_record["direction"] = direction
                setup_record["status"] = status
                setup_record["reason"] = result.get("reason")

                discovered_setups[setup_id] = setup_record

            except Exception as error:
                engine_errors.append({
                    "candle_index": end_index - 1,
                    "direction": direction,
                    "error": str(error),
                })

    status_counts = {}
    direction_counts = {
        "LONG": 0,
        "SHORT": 0,
    }

    for record in discovered_setups.values():
        status = record["status"]
        direction = record["direction"]

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        direction_counts[direction] += 1

    print("\nHISTORICAL SUMMARY")
    print("==================")
    print("Evaluated Candle Prefixes:", evaluated_prefixes)
    print("Unique Setups Discovered:", len(discovered_setups))
    print(
        "Confirmed Confluence:",
        status_counts.get("CONFIRMED_CONFLUENCE", 0),
    )
    print(
        "Watch Without Zone Overlap:",
        status_counts.get("WATCH_NO_ZONE_OVERLAP", 0),
    )
    print(
        "Other Statuses:",
        sum(
            count
            for status, count in status_counts.items()
            if status not in (
                "CONFIRMED_CONFLUENCE",
                "WATCH_NO_ZONE_OVERLAP",
            )
        ),
    )
    print("Engine Errors:", len(engine_errors))

    print("\nDIRECTION DISTRIBUTION")
    print("======================")
    print("LONG:", direction_counts["LONG"])
    print("SHORT:", direction_counts["SHORT"])

    print("\nSTATUS DISTRIBUTION")
    print("===================")

    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")

    print("\nHISTORICAL SETUP DETAILS")
    print("=======================")

    for record in discovered_setups.values():
        print("\n--------------------------------")
        print("Setup ID:", record["setup_id"])
        print("Direction:", record["direction"])
        print("Status:", record["status"])
        print("Structure Event:", record.get("structure_event"))
        print(
            "Structure Event Timestamp:",
            record.get("structure_event_timestamp"),
        )
        print(
            "Available Timestamp:",
            record.get("available_timestamp"),
        )
        print("Entry Zone:", record.get("entry_zone"))
        print("Order Block:", record.get("order_block"))
        print("FVG:", record.get("fvg"))
        print("Zone Overlap:", record.get("zone_overlap"))
        print("Stop Loss:", record.get("stop_loss"))
        print("Take Profit:", record.get("take_profit"))
        print("Reason:", record.get("reason"))

    confirmed_setups = [
        record
        for record in discovered_setups.values()
        if record.get("status") == "CONFIRMED_CONFLUENCE"
    ]

    print("\nSTRUCTURAL DEDUPLICATION")
    print("=======================")
    print(
        "Deduplication Input: Confirmed Confluence Setups Only"
    )

    deduplication_result = deduplicate_setups(
        confirmed_setups,
        default_symbol=SYMBOL,
    )

    print_deduplication_report(deduplication_result)

    print("\nRETAINED STRUCTURAL SETUPS")
    print("==========================")

    for record in deduplication_result["retained_setups"]:
        setup = record.get("setup", record)

        print("\n--------------------------------")
        print("Setup ID:", setup.get("setup_id"))
        print("Symbol:", setup.get("symbol"))
        print("Direction:", setup.get("direction"))
        print("Structure Event:", setup.get("structure_event"))
        print(
            "Structure Event Timestamp:",
            setup.get("structure_event_timestamp"),
        )
        print("Order Block:", setup.get("order_block"))
        print("FVG:", setup.get("fvg"))
        print("Entry Zone:", setup.get("entry_zone"))
        print("Stop Loss:", setup.get("stop_loss"))
        print("Take Profit:", setup.get("take_profit"))

    if engine_errors:
        print("\nENGINE ERRORS")
        print("=============")

        for error in engine_errors[:20]:
            print(error)

    print("\nHISTORICAL AUDIT COMPLETED")
    print("==========================")
    print("No trades were executed.")
    print("Live Trading: DISABLED")

    return deduplication_result


def print_setup_report(result):
    print("\n4H SETUP ENGINE REPORT")
    print("======================")

    print("Status:", result["status"])
    print("Allowed Direction:", result["allowed_direction"])
    print("Reason:", result["reason"])

    setup = result.get("setup")

    if setup is None:
        print("Setup: NONE")
        print("Entry Confirmation: NOT AVAILABLE")
        return None

    print("\nSETUP DETAILS")
    print("-------------")
    print("Setup ID:", setup["setup_id"])
    print("Symbol:", setup["symbol"])
    print("Direction:", setup["direction"])
    print("Structure Event:", setup["structure_event"])
    print("Structure Event Timestamp:", setup["structure_event_timestamp"])
    print("Available Timestamp:", setup["available_timestamp"])
    print("Expiry Timestamp:", setup["expiry_timestamp"])

    print("\nENTRY ZONE")
    print("----------")
    print("Zone Source:", setup["entry_zone"]["source"])
    print("Zone Low:", setup["entry_zone"]["low"])
    print("Zone High:", setup["entry_zone"]["high"])
    print("Reference Entry:", setup["entry_reference"])

    print("\nRISK MANAGEMENT")
    print("---------------")
    print("Stop Loss:", setup["stop_loss"])
    print("Take Profit:", setup["take_profit"])
    print("Risk Distance:", setup["risk_distance"])
    print("Risk Reward:", setup["risk_reward"])

    print("\nCONFLUENCE")
    print("----------")
    print("Order Block:", setup["order_block"])
    print("FVG:", setup["fvg"])
    print("Zone Overlap:", setup["zone_overlap"])
    print("Stop Swing:", setup["stop_swing"])

    print("Live Trading: DISABLED")

    return setup


def print_entry_confirmation_report(result):
    print("\nLOWER-TIMEFRAME ENTRY CONFIRMATION")
    print("==================================")

    if result is None:
        print("Confirmation Result: NONE")
        print("Entry Signal: NOT GENERATED")
        return

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )

    print("\nLIVE TRADING: DISABLED")


def run_lower_timeframe_confirmation(setup):
    print("\nCHECKING LOWER-TIMEFRAME CONFIRMATION")
    print("=====================================")

    if setup is None:
        print("Confirmation blocked: No valid 4H setup.")
        return None

    if setup.get("status") != "CONFIRMED_CONFLUENCE":
        print("Confirmation blocked: Setup is not confirmed.")
        print("Required status: CONFIRMED_CONFLUENCE")
        print("Current status:", setup.get("status"))
        print("Entry Signal: NOT GENERATED")
        return None

    candles_by_timeframe = {}

    for timeframe in ("1H", "15m", "5m"):
        config = TIMEFRAME_CONFIG[timeframe]

        candles = load_structure_candles(
            config["filename"],
            config["target"],
        )

        candles_by_timeframe[timeframe] = candles

        print(
            f"{timeframe}: "
            f"{len(candles)} candles loaded."
        )

    result = confirm_lower_timeframe_entry(
        setup=setup,
        candles_by_timeframe=candles_by_timeframe,
    )

    print_entry_confirmation_report(result)

    return result


def run_four_hour_setup_engine(
    market_decision,
    four_hour_candles,
    historical_deduplication=None,
):
    print("\nRUNNING 4H SETUP ENGINE")
    print("=======================")

    allowed_direction = market_decision["allowed_direction"]

    if allowed_direction == "WAIT":
        print("Setup generation blocked.")
        print("Daily and 4H market direction are not aligned.")
        return None

    result = generate_four_hour_setup(
        candles=four_hour_candles,
        allowed_direction=allowed_direction,
        setup_expiry_candles=24,
        ob_lookback=10,
        fvg_lookback=10,
        risk_reward=2.0,
        symbol=SYMBOL,
        mitigation_mode="A",
    )

    setup = result.get("setup")

    if not isinstance(setup, dict):
        print_setup_report(result)

        diagnostics = result.get("diagnostics")

        if isinstance(diagnostics, dict):
            print("\n4H SETUP DIAGNOSTICS")
            print("====================")
            print(
                json.dumps(
                    diagnostics,
                    indent=2,
                    ensure_ascii=True,
                    default=str,
                )
            )

        print("\nLOWER-TIMEFRAME CONFIRMATION SKIPPED")
        print("===================================")
        print("No 4H setup was generated.")

        return result

    setup["symbol"] = setup.get("symbol", SYMBOL)
    setup["timeframe"] = setup.get("timeframe", "4H")
    setup["status"] = result.get(
        "status",
        setup.get("status", "UNKNOWN"),
    )

    if setup["status"] == "CONFIRMED_CONFLUENCE":

        if historical_deduplication is None:
            print("\nSETUP DEDUPLICATION CHECK")
            print("=========================")
            print("Historical deduplication result is unavailable.")
            print("Entry confirmation blocked for safety.")

            result["status"] = "DEDUPLICATION_NOT_AVAILABLE"
            result["reason"] = (
                "Historical structural deduplication "
                "was not completed."
            )
            result["setup"] = None

            return result

        current_deduplication = deduplicate_setups(
            [setup],
            default_symbol=SYMBOL,
        )

        current_retained = current_deduplication.get(
            "retained_setups",
            [],
        )

        current_retained_by_key = (
            current_deduplication.get(
                "retained_by_key",
                {},
            )
        )

        if (
            not current_retained
            or len(current_retained_by_key) != 1
        ):
            print("\nSETUP DEDUPLICATION CHECK")
            print("=========================")
            print("Current setup failed structural validation.")
            print("Entry confirmation blocked.")

            result["status"] = "INVALID_STRUCTURAL_SETUP"
            result["reason"] = (
                "Current setup does not have a valid "
                "OB/FVG structural identity."
            )
            result["setup"] = None

            print_deduplication_report(
                current_deduplication,
                title="CURRENT SETUP VALIDATION",
            )

            return result

        current_key = next(
            iter(current_retained_by_key)
        )

        historical_retained_by_key = (
            historical_deduplication.get(
                "retained_by_key",
                {},
            )
        )

        historical_setup = historical_retained_by_key.get(
            current_key
        )

        if not isinstance(historical_setup, dict):
            print("\nSETUP DEDUPLICATION CHECK")
            print("=========================")
            print("No matching historical structural setup found.")
            print("Current structural key:", current_key)
            print("Entry confirmation blocked.")

            result["status"] = "HISTORICAL_SETUP_NOT_FOUND"
            result["reason"] = (
                "The current structural setup was not found "
                "in the historical deduplication registry."
            )
            result["setup"] = None

            return result

        current_available = setup.get(
            "available_timestamp"
        )

        historical_available = historical_setup.get(
            "available_timestamp"
        )

        current_event_timestamp = setup.get(
            "structure_event_timestamp"
        )

        historical_event_timestamp = historical_setup.get(
            "structure_event_timestamp"
        )

        def normalize_timestamp(value):
            if isinstance(value, bool) or value is None:
                return None

            try:
                number = float(value)
            except (TypeError, ValueError):
                return None

            if number <= 0:
                return None

            return int(round(number))

        current_available = normalize_timestamp(
            current_available
        )

        historical_available = normalize_timestamp(
            historical_available
        )

        current_event_timestamp = normalize_timestamp(
            current_event_timestamp
        )

        historical_event_timestamp = normalize_timestamp(
            historical_event_timestamp
        )

        if (
            current_available is None
            or historical_available is None
            or current_event_timestamp is None
            or historical_event_timestamp is None
        ):
            print("\nSETUP DEDUPLICATION CHECK")
            print("=========================")
            print("Setup chronology validation failed.")
            print("Entry confirmation blocked.")

            result["status"] = "INVALID_SETUP_CHRONOLOGY"
            result["reason"] = (
                "Current or historical setup timestamps "
                "are missing or invalid."
            )
            result["setup"] = None

            return result

        is_first_structural_setup = (
            current_available == historical_available
            and current_event_timestamp == historical_event_timestamp
        )

        print("\nSETUP DEDUPLICATION CHECK")
        print("=========================")
        print("Structural identity:", "VALID")
        print("Current Setup ID:", setup.get("setup_id"))
        print(
            "Historical First Setup ID:",
            historical_setup.get("setup_id"),
        )
        print(
            "Current Available Timestamp:",
            current_available,
        )
        print(
            "First Available Timestamp:",
            historical_available,
        )
        print(
            "Current Structure Event Timestamp:",
            current_event_timestamp,
        )
        print(
            "First Structure Event Timestamp:",
            historical_event_timestamp,
        )

        if not is_first_structural_setup:
            print("First retained setup: NO")
            print("Duplicate rejection: PASSED")
            print("Entry confirmation blocked.")

            result["status"] = "DUPLICATE_SETUP_REJECTED"
            result["reason"] = (
                "The current setup is not the first valid "
                "setup for its structural OB/FVG pair."
            )
            result["setup"] = None

            return result

        print("First retained setup: YES")
        print("Duplicate rejection: PASSED")

    setup = print_setup_report(result)

    if setup is None:
        print("\nLOWER-TIMEFRAME CONFIRMATION SKIPPED")
        print("===================================")
        print("No valid 4H setup is available.")
        return result

    if setup.get("status") != "CONFIRMED_CONFLUENCE":
        print("\nLOWER-TIMEFRAME CONFIRMATION SKIPPED")
        print("===================================")
        print("Setup is watch-only.")
        print("OB and FVG confluence is not confirmed.")
        print("No entry signal will be generated.")
        return result

    run_lower_timeframe_confirmation(setup)

    return result


def main():
    print("AI CRYPTO TRADING SYSTEM")
    print("========================")
    print(f"Provider: {settings.DATA_PROVIDER}")
    print(f"Market: {MARKET}")
    print(f"Symbol: {SYMBOL}")
    print("Timeframes: 1D, 4H, 1H, 15m, 5m")
    print("Mode: MULTI-TIMEFRAME STRUCTURE + ENTRY CONFIRMATION")
    print("Live Trading: DISABLED")

    if settings.TRADING_ENABLED or settings.ALLOW_LIVE_ORDERS:
        raise RuntimeError(
            "Live trading must remain disabled."
        )

    for timeframe, config in TIMEFRAME_CONFIG.items():
        initialize_archive(timeframe, config)

    for timeframe, config in TIMEFRAME_CONFIG.items():
        update_archive(timeframe, config)

    print("\nMULTI-TIMEFRAME AUDIT")
    print("=====================")

    audit_results = {}

    for timeframe, config in TIMEFRAME_CONFIG.items():
        print(f"\nTIMEFRAME: {timeframe}")

        audit_results[timeframe] = audit_archive(
            config["filename"],
            config["interval_ms"],
        )

    all_passed = all(audit_results.values())

    print("\nFINAL SYSTEM REPORT")
    print("===================")

    for timeframe, passed in audit_results.items():
        print(
            f"{timeframe}: "
            f"{'PASS' if passed else 'ISSUES DETECTED'}"
        )

    print(
        f"OVERALL AUDIT: "
        f"{'PASS' if all_passed else 'ISSUES DETECTED'}"
    )

    print("LIVE TRADING: DISABLED")

    if not all_passed:
        raise RuntimeError(
            "One or more timeframe archives failed validation."
        )

    market_decision, four_hour_candles = (
        run_market_direction_analysis()
    )

    if RUN_4H_MODE_COMPARISON_AUDIT:
        run_four_hour_mode_comparison_audit(
            four_hour_candles
        )

    historical_deduplication = None

    if RUN_HISTORICAL_CONFLUENCE_AUDIT:
        historical_deduplication = (
            run_historical_confluence_audit(
                four_hour_candles
            )
        )

    if market_decision["allowed_direction"] == "WAIT":
        print("\nSETUP GENERATION BLOCKED")
        print("========================")
        print("Market direction is not aligned.")
        print("No trade direction is currently allowed.")

    else:
        print("\nSETUP DIRECTION AUTHORIZED")
        print("==========================")
        print(
            "Authorized Direction:",
            market_decision["allowed_direction"],
        )

        run_four_hour_setup_engine(
            market_decision,
            four_hour_candles,
            historical_deduplication,
        )

    print("\nSYSTEM EXECUTION COMPLETED")
    print("==========================")
    print("No trades were executed.")
    print("Live Trading: DISABLED")


if __name__ == "__main__":
    main()
