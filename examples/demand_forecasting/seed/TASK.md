# Demand forecasting task

Improve `solution/train.py` to forecast daily demand.

## Runtime contract

The evaluator sets:

- `AUTORESEARCH_TRAIN_DATA`: Parquet training history with `sku`, `date`, `demand`,
  `promo`, `price`, and `category`.
- `AUTORESEARCH_REQUEST`: Parquet containing exactly the `sku` and future `date` rows to forecast.
- `AUTORESEARCH_OUTPUT`: destination path.

Write a Parquet file to `AUTORESEARCH_OUTPUT` containing exactly `sku`, `date`, and `forecast`.
Every requested row must appear once. Forecasts must be finite and non-negative.

Only edit files under `solution/`. The primary metric is WMAPE (lower is better). Guardrails
track MAPE, RMSE, aggregate bias, runtime, and a hidden later holdout. The process has a 10-minute
wall-clock limit, runs on a laptop, and cannot inspect validation or holdout actuals.

You may derive local chronological validation splits from the training history. Prefer a focused,
testable experiment over broad rewrites.
