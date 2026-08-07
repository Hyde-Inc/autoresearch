# retail_sales_sample demand forecasting

Improve `solution/train.py` to forecast `sales` per `sku_name` and `date`.

## Runtime contract

The evaluator sets:

- `AUTORESEARCH_TRAIN_DATA`: Parquet training history with columns `date`, `sku_name`, `sales`, `selling_price`.
- `AUTORESEARCH_REQUEST`: Parquet with exactly the `sku_name` and future `date`
  rows to forecast.
- `AUTORESEARCH_OUTPUT`: destination path.

Write a Parquet file to `AUTORESEARCH_OUTPUT` with exactly `sku_name`, `date`,
and `forecast`. Every requested row must appear once. Forecasts must be finite and
non-negative.

Only edit files under `solution/`. You cannot see validation or holdout actuals. Prefer a
focused, testable experiment over broad rewrites.
