
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


MARKET = "SPOT"
SYMBOL = "BTC-USDT"

RUN_HISTORICAL_CONFLUENCE_AUDIT = True

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

                discovered_setups[setup_id] = {
                    "setup_id": setup_id,
                    "direction": direction,
                    "status": status,
                    "structure_event": setup.get(
                        "structure_event"
                    ),
                    "structure_event_timestamp": setup.get(
                        "structure_event_timestamp"
                    ),
                    "available_timestamp": available_timestamp,
                    "entry_zone": setup.get("entry_zone"),
                    "order_block": setup.get("order_block"),
                    "fvg": setup.get("fvg"),
                    "zone_overlap": setup.get("zone_overlap"),
                    "stop_loss": setup.get("stop_loss"),
                    "take_profit": setup.get("take_profit"),
                    "reason": result.get("reason"),
                }

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
        print("Structure Event:", record["structure_event"])
        print(
            "Structure Event Timestamp:",
            record["structure_event_timestamp"],
        )
        print(
            "Available Timestamp:",
            record["available_timestamp"],
        )
        print("Entry Zone:", record["entry_zone"])
        print("Order Block:", record["order_block"])
        print("FVG:", record["fvg"])
        print("Zone Overlap:", record["zone_overlap"])
        print("Stop Loss:", record["stop_loss"])
        print("Take Profit:", record["take_profit"])
        print("Reason:", record["reason"])

    if engine_errors:
        print("\nENGINE ERRORS")
        print("=============")

        for error in engine_errors[:20]:
            print(error)

    print("\nHISTORICAL AUDIT COMPLETED")
    print("==========================")
    print("No trades were executed.")
    print("Live Trading: DISABLED")

    return discovered_setups


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
    )

    setup = print_setup_report(result)

    if setup is None:
        print("\nLOWER-TIMEFRAME CONFIRMATION SKIPPED")
        print("===================================")
        print("No 4H setup was generated.")
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

    if RUN_HISTORICAL_CONFLUENCE_AUDIT:
        run_historical_confluence_audit(
            four_hour_candles
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
        )

    print("\nSYSTEM EXECUTION COMPLETED")
    print("==========================")
    print("Live Trading: DISABLED")


if __name__ == "__main__":
    main()
