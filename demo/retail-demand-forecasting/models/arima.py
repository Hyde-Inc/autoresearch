"""Candidate forecaster: per-SKU ARIMA(1,0,1) with weekday regressors.

The data science team built this as an upgrade over the seasonal-naive model.
Backtests on the final 28 days of history, then writes the next 28 days of
forecasts for the replenishment system.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sales_daily.parquet"
OUTPUT = ROOT / "forecasts" / "next_28_days_arima.csv"
HORIZON = 28

warnings.filterwarnings("ignore")  # statsmodels convergence chatter


def weekday_dummies(dates: pd.DatetimeIndex) -> np.ndarray:
    """Six weekday indicator columns (Monday is the reference level)."""
    return np.column_stack([(dates.dayofweek == day).astype(float) for day in range(1, 7)])


def fit_and_forecast(series: pd.Series, request_dates: pd.DatetimeIndex) -> np.ndarray:
    """ARIMA(1,0,1) with weekday regressors; falls back to the recent mean."""
    fallback = float(series.tail(28).mean())
    if series.sum() == 0 or series.nunique() < 3:
        return np.full(len(request_dates), max(0.0, fallback))
    try:
        model = ARIMA(
            series.to_numpy(dtype=float),
            exog=weekday_dummies(pd.DatetimeIndex(series.index)),
            order=(1, 0, 1),
        ).fit(method_kwargs={"maxiter": 50})
        forecast = model.forecast(len(request_dates), exog=weekday_dummies(request_dates))
        return np.clip(np.nan_to_num(forecast, nan=fallback), 0.0, None)
    except Exception:
        return np.full(len(request_dates), max(0.0, fallback))


def run(history: pd.DataFrame, request_dates: pd.DatetimeIndex) -> pd.DataFrame:
    forecasts = []
    for sku, group in history.groupby("sku_id", sort=False):
        series = group.set_index("date")["units_sold"].sort_index().asfreq("D").fillna(0.0)
        values = fit_and_forecast(series, request_dates)
        forecasts.append(
            pd.DataFrame({"sku_id": sku, "date": request_dates, "forecast": values})
        )
    return pd.concat(forecasts, ignore_index=True)


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
    request = pd.DatetimeIndex(sorted(actuals["date"].unique()))
    backtest = run(history, request)
    merged = actuals.merge(backtest, on=["sku_id", "date"])
    score = wmape(merged["units_sold"].to_numpy(), merged["forecast"].to_numpy())
    print(f"Backtest WMAPE over the last {HORIZON} days: {score:.4f}")

    # Production forecast: the next HORIZON days after the data ends.
    future = pd.date_range(last_day + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    forecast = run(sales, future)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(forecast):,} forecast rows to {OUTPUT}")


if __name__ == "__main__":
    main()
