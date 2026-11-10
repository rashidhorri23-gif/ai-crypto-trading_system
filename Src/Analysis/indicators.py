
def calculate_vwap(data):
    typical_price = (
        data["high"]
        + data["low"]
        + data["close"]
    ) / 3

    utc_days = pd.to_datetime(
        data["timestamp"],
        unit="ms",
        utc=True
    ).dt.floor("D")

    cumulative_volume = (
        data["volume"]
        .groupby(utc_days, sort=False)
        .cumsum()
    )

    cumulative_value = (
        (typical_price * data["volume"])
        .groupby(utc_days, sort=False)
        .cumsum()
    )

    return (
        cumulative_value
        / cumulative_volume.replace(0, np.nan)
    )
