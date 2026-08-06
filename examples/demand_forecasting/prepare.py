"""Create a deterministic, small retail demand forecasting benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 42
N_SKUS = 60
N_DAYS = 730
VAL_DAYS = 28
HOLDOUT_DAYS = 28


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    rows: list[pd.DataFrame] = []
    for index in range(N_SKUS):
        base = rng.uniform(8, 80)
        weekly_amp = rng.uniform(0.08, 0.35)
        annual_amp = rng.uniform(0.03, 0.20)
        trend = rng.uniform(-0.00025, 0.0008)
        phase = rng.uniform(0, 2 * np.pi)
        day = np.arange(N_DAYS)
        promo = rng.random(N_DAYS) < rng.uniform(0.02, 0.08)
        price = rng.uniform(5, 100) * (1 - promo * rng.uniform(0.05, 0.25))
        signal = (
            base
            * (1 + trend * day)
            * (1 + weekly_amp * np.sin(2 * np.pi * day / 7 + phase))
            * (1 + annual_amp * np.sin(2 * np.pi * day / 365 + phase / 2))
            * (1 + promo * rng.uniform(0.15, 0.65))
        )
        noise = rng.normal(0, np.sqrt(np.maximum(signal, 1)) * 0.8)
        demand = np.maximum(0, np.round(signal + noise))
        rows.append(
            pd.DataFrame(
                {
                    "sku": f"SKU_{index:03d}",
                    "date": dates,
                    "demand": demand.astype(float),
                    "promo": promo.astype(int),
                    "price": price,
                    "category": f"CAT_{index % 8:02d}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    frame = build()
    train_end = N_DAYS - VAL_DAYS - HOLDOUT_DAYS
    dates = sorted(frame["date"].unique())
    train = frame[frame["date"].isin(dates[:train_end])]
    validation = frame[frame["date"].isin(dates[train_end : train_end + VAL_DAYS])]
    holdout = frame[frame["date"].isin(dates[train_end + VAL_DAYS :])]
    train_path = ROOT / "seed/data/train.parquet"
    private = ROOT / "private"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    validation[["sku", "date", "demand"]].to_parquet(
        private / "validation.parquet", index=False
    )
    holdout[["sku", "date", "demand"]].to_parquet(private / "holdout.parquet", index=False)
    print(
        f"Created {len(train):,} train rows, {len(validation):,} validation rows, "
        f"and {len(holdout):,} hidden holdout rows."
    )


if __name__ == "__main__":
    main()
