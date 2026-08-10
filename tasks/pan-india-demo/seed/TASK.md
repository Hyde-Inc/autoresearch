# pan-india-demo demand forecasting

Improve `solution/train.py` to forecast `sales` per `sku_name` and `date`.

## Runtime contract

The evaluator sets:

- `AUTORESEARCH_TRAIN_DATA`: Parquet training history with columns `sku_name`, `sh`, `date`, `vertical`, `sales`, `adjusted_units`, `selling_price`, `mean_mrp`, `last_30_days_mean_fsp`, `price_index_fsp_based`, `price_index_mrp_based`, `past15daysmean`, `pe_flag`, `oos_flag`, `normal_day_flag`, `sales_filter_flag`, `midnight_oos_stock`, `morning_oos_stock`, `afternoon_oos_stock`, `evening_oos_stock`, `night_oos_stock`, `midnight_drr`, `morning_drr`, `afternoon_drr`, `evening_drr`, `night_drr`, `fgp_date`, `lower_bound`, `upper_bound`, `expected_sales`, `final_adjusted_units`, `expected_sales_per_unit`, `cap_bind_flag`.
- `AUTORESEARCH_REQUEST`: Parquet with exactly the `sku_name` and future `date`
  rows to forecast.
- `AUTORESEARCH_OUTPUT`: destination path.

Write a Parquet file to `AUTORESEARCH_OUTPUT` with exactly `sku_name`, `date`,
and `forecast`. Every requested row must appear once. Forecasts must be finite and
non-negative.

Only edit files under `solution/`. You cannot see validation or holdout actuals. Prefer a
focused, testable experiment over broad rewrites.
