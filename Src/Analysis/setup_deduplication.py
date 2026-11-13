
from collections import defaultdict
from datetime import datetime, timezone


def _get_setup(record):
    if not isinstance(record, dict):
        return None

    nested_setup = record.get("setup")

    if isinstance(nested_setup, dict):
        return nested_setup

    return record


def _to_timestamp(value):
    if isinstance(value, bool) or value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number <= 0:
        return None

    return int(round(number))


def _get_structural_key(record, default_symbol="BTC-USDT"):
    setup = _get_setup(record)

    if setup is None:
        return None, "INVALID_SETUP_RECORD"

    order_block = setup.get("order_block")
    fvg = setup.get("fvg")

    if not isinstance(order_block, dict):
        return None, "MISSING_ORDER_BLOCK"

    if not isinstance(fvg, dict):
        return None, "MISSING_FVG"

    symbol = str(
        setup.get("symbol", default_symbol)
    ).strip().upper()

    timeframe = str(
        setup.get("timeframe", "4H")
    ).strip().upper()

    direction = str(
        setup.get("direction", "")
    ).strip().upper()

    ob_timestamp = _to_timestamp(
        order_block.get("timestamp")
    )

    fvg_timestamp = _to_timestamp(
        fvg.get("timestamp")
    )

    if not symbol:
        return None, "MISSING_SYMBOL"

    if not timeframe:
        return None, "MISSING_TIMEFRAME"

    if direction not in ("LONG", "SHORT"):
        return None, "INVALID_DIRECTION"

    if ob_timestamp is None:
        return None, "INVALID_OB_TIMESTAMP"

    if fvg_timestamp is None:
        return None, "INVALID_FVG_TIMESTAMP"

    return (
        symbol,
        timeframe,
        direction,
        ob_timestamp,
        fvg_timestamp,
    ), None


def _get_sort_timestamp(record, field, fallback=None):
    setup = _get_setup(record)

    if setup is None:
        return None

    value = setup.get(field)

    if value is None and fallback:
        value = setup.get(fallback)

    return _to_timestamp(value)


def _format_timestamp(value):
    if value is None:
        return "N/A"

    timestamp = value / 1000 if value > 100000000000 else value

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def deduplicate_setups(
    setups,
    default_symbol="BTC-USDT",
):
    if not isinstance(setups, list):
        raise TypeError("setups must be a list.")

    valid_records = []
    invalid_setups = []

    for original_index, record in enumerate(setups):
        key, error = _get_structural_key(
            record,
            default_symbol=default_symbol,
        )

        if error:
            invalid_setups.append({
                "record": record,
                "original_index": original_index,
                "reason": error,
            })
            continue

        available_timestamp = _get_sort_timestamp(
            record,
            "available_timestamp",
        )

        if available_timestamp is None:
            invalid_setups.append({
                "record": record,
                "original_index": original_index,
                "reason": "INVALID_AVAILABLE_TIMESTAMP",
            })
            continue

        event_timestamp = _get_sort_timestamp(
            record,
            "structure_event_timestamp",
            fallback="timestamp",
        )

        if event_timestamp is None:
            event_timestamp = available_timestamp

        valid_records.append({
            "record": record,
            "key": key,
            "available_timestamp": available_timestamp,
            "event_timestamp": event_timestamp,
            "original_index": original_index,
        })

    valid_records.sort(
        key=lambda item: (
            item["available_timestamp"],
            item["event_timestamp"],
            item["original_index"],
        )
    )

    retained_setups = []
    duplicate_setups = []
    retained_by_key = {}
    duplicate_groups = defaultdict(list)

    for item in valid_records:
        key = item["key"]
        record = item["record"]

        if key not in retained_by_key:
            retained_by_key[key] = record
            retained_setups.append(record)
            duplicate_groups[key].append(record)
        else:
            first_record = retained_by_key[key]

            duplicate_setups.append({
                "record": record,
                "first_record": first_record,
                "key": key,
            })

            duplicate_groups[key].append(record)

    duplicate_groups = {
        key: records
        for key, records in duplicate_groups.items()
        if len(records) > 1
    }

    return {
        "total_input": len(setups),
        "valid_input": len(valid_records),
        "retained_setups": retained_setups,
        "duplicate_setups": duplicate_setups,
        "invalid_setups": invalid_setups,
        "duplicate_groups": duplicate_groups,
        "retained_by_key": retained_by_key,
    }


def print_deduplication_report(
    result,
    title="4H SETUP DEDUPLICATION REPORT",
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print("Total input records:", result["total_input"])
    print("Valid records:", result["valid_input"])
    print("Retained unique setups:", len(result["retained_setups"]))
    print("Duplicate setups:", len(result["duplicate_setups"]))
    print("Invalid setups:", len(result["invalid_setups"]))
    print("Duplicate structural groups:", len(result["duplicate_groups"]))

    print()
    print("-" * 72)
    print("DUPLICATE SETUP DETAILS")
    print("-" * 72)

    if not result["duplicate_setups"]:
        print("No duplicate setups found.")

    for number, item in enumerate(
        result["duplicate_setups"],
        start=1,
    ):
        duplicate = _get_setup(item["record"])
        first = _get_setup(item["first_record"])

        print()
        print(f"Duplicate #{number}")
        print("Structural Key:", item["key"])
        print("Retained Setup ID:", first.get("setup_id"))
        print("Duplicate Setup ID:", duplicate.get("setup_id"))
        print(
            "Retained Available:",
            _format_timestamp(
                _get_sort_timestamp(
                    item["first_record"],
                    "available_timestamp",
                )
            ),
        )
        print(
            "Duplicate Available:",
            _format_timestamp(
                _get_sort_timestamp(
                    item["record"],
                    "available_timestamp",
                )
            ),
        )

    print()
    print("-" * 72)
    print("INVALID SETUP DETAILS")
    print("-" * 72)

    if not result["invalid_setups"]:
        print("No invalid setup records.")

    for number, item in enumerate(
        result["invalid_setups"],
        start=1,
    ):
        print(
            f"Invalid #{number} | "
            f"Index: {item['original_index']} | "
            f"Reason: {item['reason']}"
        )

    print()
    print("=" * 72)
    print("DEDUPLICATION COMPLETED")
    print("=" * 72)
