
import os

import numpy as np

from .config import Config
from .market_features import prepare_market_features


def main():
    print("=" * 70)
    print("MARKET FEATURES VALIDATION TEST")
    print("=" * 70)

    file_path = os.path.join(
        Config.DATA_DIRECTORY,
        "BTC_USDT_SPOT_4H.csv",
    )

    print(f"Data file: {file_path}")

    if not os.path.exists(file_path):
        print("ERROR: Market data file not found.")
        return 1

    try:
        df = prepare_market_features(file_path)

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}"
        )
        return 1

    failed = False

    print(f"\nTotal candles: {len(df)}")

    print("\nLatest candle:")
    columns = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "adx_14",
        "atr_14",
        "plus_di",
        "minus_di",
        "relative_volume",
        "support",
        "resistance",
        "market_structure",
        "ema_trend",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "macd",
        "macd_signal",
        "macd_histogram",
        "vwap_utc",
        "choch_bullish",
        "choch_bearish",
        "structure_bias",
        "liquidity_sweep_high",
        "liquidity_sweep_low",
        "fvg_bullish_low",
        "fvg_bullish_high",
        "fvg_bearish_low",
        "fvg_bearish_high",
        "last_bullish_ob_low",
        "last_bullish_ob_high",
        "last_bearish_ob_low",
        "last_bearish_ob_high",
    ]

    print(
        df[columns].tail(1).to_string(
            index=False
        )
    )

    print("\nFeature validation:")

    feature_columns = [
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "adx_14",
        "atr_14",
        "plus_di",
        "minus_di",
        "relative_volume",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "macd",
        "macd_signal",
        "macd_histogram",
        "vwap_utc",
    ]

    for column in feature_columns:
        valid_count = int(
            df[column].notna().sum()
        )

        invalid_count = int(
            np.isinf(
                df[column].dropna()
            ).sum()
        )

        passed = (
            valid_count > 0
            and invalid_count == 0
        )

        print(
            f"{column}: "
            f"{'PASS' if passed else 'FAIL'} "
            f"| Valid: {valid_count} "
            f"| Infinite: {invalid_count}"
        )

        if not passed:
            failed = True

    print("\nIndicator range validation:")

    rsi_valid = df["rsi_14"].dropna()

    rsi_passed = (
        not rsi_valid.empty
        and rsi_valid.between(0, 100).all()
    )

    print(
        f"RSI range 0-100: "
        f"{'PASS' if rsi_passed else 'FAIL'}"
    )

    if not rsi_passed:
        failed = True

    atr_valid = df["atr_14"].dropna()

    atr_passed = (
        not atr_valid.empty
        and (atr_valid > 0).all()
    )

    print(
        f"ATR positive values: "
        f"{'PASS' if atr_passed else 'FAIL'}"
    )

    if not atr_passed:
        failed = True

    print("\nMarket data validation:")

    expected_interval = 4 * 60 * 60 * 1000

    timestamp_differences = (
        df["timestamp"].diff().dropna()
    )

    irregular_intervals = int(
        (timestamp_differences != expected_interval).sum()
    )

    print(
        f"Expected 4H intervals: "
        f"{'PASS' if irregular_intervals == 0 else 'FAIL'}"
    )

    print(
        f"Irregular intervals: {irregular_intervals}"
    )

    if irregular_intervals > 0:
        failed = True

    print(
        f"Duplicate timestamps: "
        f"{'FAIL' if df['timestamp'].duplicated().any() else 'PASS'}"
    )

    if df["timestamp"].duplicated().any():
        failed = True

    print("\nSwing structure validation:")

    swing_high_count = int(
        df["swing_high"].notna().sum()
    )

    swing_low_count = int(
        df["swing_low"].notna().sum()
    )

    print(f"Confirmed swing highs: {swing_high_count}")
    print(f"Confirmed swing lows: {swing_low_count}")

    structure_passed = (
        swing_high_count > 0
        and swing_low_count > 0
    )

    print(
        f"Swing detection: "
        f"{'PASS' if structure_passed else 'FAIL'}"
    )

    if not structure_passed:
        failed = True

    print("\nFINAL RESULT:")

    if failed:
        print("MARKET FEATURES TEST FAILED")
        return 1

    print("MARKET FEATURES TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
