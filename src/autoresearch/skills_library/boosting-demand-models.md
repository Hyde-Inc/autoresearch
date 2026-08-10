---
name: boosting-demand-models
description: "Implement and debug global demand forecasters with XGBoost, LightGBM, or CatBoost: leakage-safe panel features, direct multi-horizon training, objectives, categorical handling, early stopping, and runtime-safe tuning."
---
# Boosting models for panel demand

Use this skill when many related items share calendar, price, promotion, or
lag effects. A single global model usually has more useful training rows than
one tree model per item. Do not use it merely because boosting is available:
first beat seasonal naive on a horizon-matched temporal backtest.

## Inspect the environment before editing

Read `pyproject.toml` or the lockfile and use an installed backend. Do not spend
an agent session installing and benchmarking three libraries. Smoke-test the
import and version:

```python
import importlib.metadata as md

for package in ("xgboost", "lightgbm", "catboost"):
    try:
        print(package, md.version(package))
    except md.PackageNotFoundError:
        pass
```

Choose one backend:

- **XGBoost**: strong default when it is already installed; `tree_method="hist"`
  is the practical CPU path. Native pandas categoricals require category dtype,
  `enable_categorical=True`, and a supported tree method (`hist` or `approx`).
- **LightGBM**: often fastest on wide tabular panels. Use pandas categorical
  columns or contiguous non-negative integer category codes; unknown categories
  become missing. Use callback-based early stopping.
- **CatBoost**: useful for many high-cardinality item/category fields and minimal
  encoding. Pass categorical column names through `cat_features`; never feed
  target-derived encodings as ordinary categoricals.

## Build one reproducible supervised panel

For every row, define a forecast origin and horizon `h`. The target is
`y[item, origin + h]`; all target-derived features must stop at `origin`.
For a direct model covering horizons `1..H`, stack all `(item, origin, h)` rows
and include `h` as a feature. This yields one model and no recursive feedback.

Demand features:

```python
g = frame.sort_values(["sku_id", "date"]).groupby("sku_id")["units_sold"]
for lag in (7, 14, 28, 35):
    frame[f"lag_{lag}"] = g.shift(lag)
for window in (7, 14, 28):
    shifted = g.shift(1)
    frame[f"mean_{window}"] = (
        shifted.groupby(frame["sku_id"]).rolling(window).mean()
        .reset_index(level=0, drop=True)
    )
```

That snippet is valid for one-step rows. For a direct `h`-step row, a feature
at calendar date `origin + h` may still only read demand through `origin`;
generate from explicit origins rather than shifting the final prediction table
and hoping it is safe.

Add only information known over the complete forecast horizon:

- target-date day-of-week, week/month, holiday flags;
- stable item/category identity;
- planned promotion, price, discount depth, and promo age when those plans
  really are available at inference;
- item-relative price (`price / shifted_item_median_price`) and shifted price
  changes;
- age, days since last nonzero sale, zero share, and recent demand volatility.

Never use unshifted target means, target encodings computed on the full frame,
future observed prices, centered rolling windows, or a random train/test split.
Build the exact requested rows in a smoke test and assert every feature is
finite or deliberately missing.

## Match the objective to the target

- Non-negative continuous/count demand with many exact zeros: try Tweedie on
  the raw target. In XGBoost use `objective="reg:tweedie"` and
  `tweedie_variance_power` strictly between 1 and 2. In LightGBM use
  `objective="tweedie"` and the same parameter. Values near 1 are more
  Poisson-like; values near 2 are more Gamma-like. Test a small set such as
  `{1.1, 1.3, 1.5}`, not a long sweep.
- Smooth demand: squared error, absolute error, or Huber can be stronger than
  Tweedie. A `log1p(y)` target is an experiment, not a default; invert with
  `expm1`, clip at zero, and check negative bias introduced by Jensen's
  inequality.
- Asymmetric inventory cost: use quantile loss when the backend supports it;
  the requested quantile must come from the business loss, not arbitrary
  optimism.

The training objective need not equal WMAPE. Select with the protected metric
on out-of-sample predictions, and always inspect signed bias.

## Concrete backend patterns

XGBoost sklearn API:

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    objective="reg:tweedie",
    tweedie_variance_power=1.3,
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
```

Check the installed XGBoost version before relying on sklearn early-stopping
arguments; the stable cross-version escape hatch is `xgb.train()` with
`DMatrix`/`QuantileDMatrix` and `early_stopping_rounds`.

LightGBM native API:

```python
import lightgbm as lgb

train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_columns)
valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set,
                        categorical_feature=cat_columns)
model = lgb.train(
    {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    },
    train_set,
    num_boost_round=2000,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
forecast = model.predict(X_future, num_iteration=model.best_iteration)
```

CatBoost pattern:

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.3",
    iterations=2000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=10,
    random_seed=42,
    verbose=False,
)
model.fit(X_train, y_train, cat_features=cat_columns,
          eval_set=(X_valid, y_valid), early_stopping_rounds=100)
```

Confirm the installed CatBoost build supports the chosen loss on the selected
device. Fall back to RMSE/MAE rather than improvising an untested custom loss.

## Tune in the right order

1. Prove seasonal-naive parity with only item, calendar, and seasonal lags.
2. Add rolling features.
3. Add known-future price/promo features.
4. Compare one or two objectives.
5. Then adjust capacity (`max_depth`/`num_leaves`, minimum leaf size,
   learning rate and rounds).

Use early stopping on the most recent horizon-sized origin, then verify the
chosen configuration on at least one earlier rolling origin when the runtime
budget allows. Persist only the final implementation in `solution/`; temporary
sweeps must not become the production path.

## Required checks before reporting

- output covers every requested item/date exactly once;
- forecasts are finite and clipped to physical bounds;
- category mappings are stable between train and request rows;
- no demand observed after each origin enters any feature;
- metric and bias are reported overall, by horizon, and for sparse items;
- runtime includes feature construction and inference, not training alone;
- seasonal-naive fallback handles unseen/short-history items.
