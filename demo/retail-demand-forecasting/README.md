# Retail Demand Forecasting

Daily demand forecasting for the retail replenishment team. We forecast unit
sales per SKU 28 days ahead; the forecasts drive store ordering, so systematic
over-forecasting creates waste and under-forecasting creates stockouts.

## Data

`data/sales_daily.parquet` holds 32 weeks of daily sales for 8 SKUs (1,792
rows). A fresh autoresearch setup reserves two 28-day windows for validation
and hidden holdout, leaving 24 weeks for model training:

| column     | meaning                                    |
| ---------- | ------------------------------------------ |
| sku_id     | product identifier                         |
| date       | calendar day                               |
| units_sold | units sold that day (the forecast target)  |
| promo      | 1 when the SKU was on promotion            |
| price      | shelf price that day                       |
| category   | product category                           |

The dataset is synthetic for this demo. Regenerate it with:

```bash
uv run python scripts/make_dataset.py
```

Roughly a quarter of the SKUs are slow movers with intermittent demand (many
zero-sale days); the rest have strong weekly seasonality and promo uplift.

## Current models

`models/seasonal_baseline.py` is the model currently used in production: a
seasonal-naive forecaster that repeats the same weekday from the most recent
week of history. `models/arima.py` is the team's candidate upgrade: per-SKU
ARIMA(1,0,1) with weekday regressors. Both backtest on the last 28 days and
write next month's forecast:

```bash
uv run python models/seasonal_baseline.py
uv run python models/arima.py
```

## Project layout

```
data/       raw sales history (generated, not committed)
models/     production forecasting model
scripts/    dataset utilities
forecasts/  model output (not committed)
```
