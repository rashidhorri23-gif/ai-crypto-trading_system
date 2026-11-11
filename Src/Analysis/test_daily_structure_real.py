def format_timestamp(timestamp):
    from datetime import datetime, timezone

    timestamp = int(float(timestamp))

    if timestamp > 100000000000:
        timestamp = timestamp / 1000

    return datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
