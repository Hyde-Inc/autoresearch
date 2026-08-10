"""Generate the synthetic daily sales history for this demo project.

Kept intentionally tiny (8 SKUs x 32 weeks, 1,792 rows) so repeated protected
training and evaluation stays quick during a live demo, while preserving
weekly and longer seasonality, promo uplift, price changes, and a quarter of
SKUs with intermittent demand.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SKUS = 8
N_INTERMITTENT = 2
N_DAYS = 224


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    day = np.arange(N_DAYS)
    rows: list[pd.DataFrame] = []
    for index in range(N_SKUS):
        intermittent = index < N_INTERMITTENT
        promo = rng.random(N_DAYS) < rng.uniform(0.02, 0.08)
        price = rng.uniform(5, 100) * (1 - promo * rng.uniform(0.05, 0.25))
        if intermittent:
            # Slow movers: rare purchase days with small basket sizes.
            buy_probability = rng.uniform(0.05, 0.25)
            demand = np.where(
                rng.random(N_DAYS) < buy_probability * (1 + promo * 0.8),
                rng.poisson(rng.uniform(1, 4), N_DAYS),
                0,
            ).astype(float)
        else:
            base = rng.uniform(8, 80)
            weekly_amp = rng.uniform(0.08, 0.35)
            annual_amp = rng.uniform(0.03, 0.20)
            trend = rng.uniform(-0.00025, 0.0008)
            phase = rng.uniform(0, 2 * np.pi)
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
                    "sku_id": f"SKU_{index:03d}",
                    "date": dates,
                    "units_sold": demand.astype(float),
                    "promo": promo.astype(int),
                    "price": price.round(2),
                    "category": f"CAT_{index % 6:02d}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    frame = build()
    target = ROOT / "data" / "sales_daily.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    zero_share = float((frame["units_sold"] == 0).mean())
    print(
        f"Wrote {len(frame):,} rows for {frame['sku_id'].nunique()} SKUs to {target} "
        f"(zero-sale share {zero_share:.1%})."
    )


if __name__ == "__main__":
    main()
