"""Production forecaster: seasonal-naive, repeating the same weekday last week.

Backtests on the final 28 days of history, then writes the next 28 days of
forecasts for the replenishment system.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sales_daily.parquet"
OUTPUT = ROOT / "forecasts" / "next_28_days.csv"
HORIZON = 28


def seasonal_naive(history: pd.DataFrame, request_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One forecast row per SKU/date: the value from the same weekday a week earlier."""
    forecasts = []
    for sku, group in history.groupby("sku_id", sort=False):
        series = group.set_index("date")["units_sold"].sort_index()
        fallback = float(series.tail(28).mean())
        for date in request_dates:
            lag = date - pd.Timedelta(days=7)
            steps = 0
            while lag not in series.index and steps < 8:
                lag -= pd.Timedelta(days=7)
                steps += 1
            value = float(series.get(lag, fallback))
            forecasts.append({"sku_id": sku, "date": date, "forecast": max(0.0, value)})
    return pd.DataFrame(forecasts)


def wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(forecast - actual).sum() / np.abs(actual).sum())


def main() -> None:
    sales = pd.read_parquet(DATA)
    sales["date"] = pd.to_datetime(sales["date"])
    last_day = sales["date"].max()

    # Backtest: hold out the final HORIZON days.
    cutoff = last_day - pd.Timedelta(days=HORIZON)
    history = sales[sales["date"] <= cutoff]
    actuals = sales[sales["date"] > cutoff]
    backtest = seasonal_naive(history, pd.DatetimeIndex(sorted(actuals["date"].unique())))
    merged = actuals.merge(backtest, on=["sku_id", "date"])
    score = wmape(merged["units_sold"].to_numpy(), merged["forecast"].to_numpy())
    print(f"Backtest WMAPE over the last {HORIZON} days: {score:.4f}")

    # Production forecast: the next HORIZON days after the data ends.
    future = pd.date_range(last_day + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    forecast = seasonal_naive(sales, future)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(forecast):,} forecast rows to {OUTPUT}")


if __name__ == "__main__":
    main()
