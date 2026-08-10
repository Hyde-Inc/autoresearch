---
name: temporal-validation-and-leakage
description: "Design and audit demand-forecast backtests that match production: rolling origins, multi-horizon feature availability, panel splits, hidden-holdout hygiene, metric decomposition, and explicit leakage tests."
---
# Temporal validation and leakage control

Use this skill when building any supervised forecast dataset, when validation
improves but holdout regresses, or when a result looks implausibly strong.
Forecasting correctness is primarily an information-availability problem:
every prediction row must use only data available at its forecast origin.

## Match the production contract

Write down:

- forecast origin;
- requested dates and horizon `H`;
- whether forecasts are produced once for all `H` dates or refreshed each day;
- columns genuinely known for all future dates;
- item/date coverage and fallback behavior;
- promotion metric and runtime guardrails.

A 28-day one-shot forecast must be backtested as a 28-day one-shot forecast.
One-step predictions rolled across the same period answer a different,
easier question.

## Split by time, never rows

For a panel, all items share each origin:

```python
dates = np.sort(frame["date"].unique())
valid_dates = dates[-H:]
train = frame[frame["date"] < valid_dates[0]]
valid = frame[frame["date"].isin(valid_dates)]
```

Never random-split rows, because adjacent rows from the same item share target
history and temporal regime. Never let one item's future dates enter training
while another item is being validated at those dates unless production truly
has that information.

Use multiple rolling origins when data and runtime allow. Keep the final hidden
holdout untouched by feature selection, hyperparameter tuning, calibration,
router thresholds, and blend weights. A holdout loss may inform a future
hypothesis about drift, but its values must never be fed back to the same
experiment loop.

## Audit feature availability by origin

Classify each field:

- **past-observed**: demand, realized sales, stock, actual price;
- **static**: item/category/store identity;
- **deterministic future**: date, weekday, planned holiday;
- **known future only if supplied**: promotions, planned price, availability;
- **never known**: future demand and statistics containing it.

Common leakage:

- unshifted rolling/expanding target aggregates;
- target encoding calculated on full data;
- centered windows;
- interpolation that uses both sides of a missing point;
- normalizers fit on validation;
- "actual price" columns whose future values are not planned prices;
- recursive features accidentally refreshed with validation actuals;
- selecting a router/model using hidden holdout metrics.

For direct horizon rows, test the strongest condition: replacing every demand
value after `origin` with a sentinel must not change any feature for any
`origin + h` prediction.

```python
def assert_origin_safe(build_features, history, origin, future_rows):
    original = build_features(history, origin, future_rows)
    poisoned = history.copy()
    poisoned.loc[poisoned["date"] > origin, "units_sold"] = 1e12
    candidate = build_features(poisoned, origin, future_rows)
    pd.testing.assert_frame_equal(original, candidate)
```

## Direct versus recursive validation

**Direct**: train one model per horizon or one stacked model with `h`; target
lags are anchored at origin. Validate all horizons together.

**Recursive**: predict step 1, append that forecast, recompute features, and
repeat. The backtest must feed predictions—not actuals—back into later steps.
Measure error by horizon because compounding can hide behind aggregate WMAPE.

**Statistical direct forecast**: fit once at the origin and call
`forecast(steps=H)`. Do not refit after observing each validation day.

## Evaluate beyond one aggregate

Use the configured metric for promotion, then decompose out-of-sample errors:

- signed bias overall and by item;
- contribution to absolute error by item/demand scale;
- metric by horizon bucket and weekday;
- sparse versus dense items;
- promotion versus ordinary days;
- fallback versus primary model;
- runtime and failure counts.

WMAPE weights high-volume observations and can hide failures on slow movers.
MAPE is unstable at zero. RMSE emphasizes spikes. These are diagnostics, not
replacements for the task metric.

## Diagnose validation/holdout gaps

A validation win followed by a later holdout loss usually indicates:

- tuning to one origin;
- level/promo/zero-rate regime change;
- unstable per-item parameters or calibration;
- leakage tied to the validation construction;
- gains concentrated in a few dates/items.

Respond by testing earlier rolling origins, simplifying the model, shortening
or weighting stale history, or shrinking segment-specific choices. Do not use
the holdout to choose the correction.

## Required tests before reporting

- exact item/date output coverage and no duplicates;
- origin-poison leakage test for target-derived features;
- fitted preprocessors see training rows only;
- recursive backtest feeds predictions back;
- known-future covariates are available in the real request;
- one clean seasonal-naive comparison on identical origins;
- primary metric, bias, horizon breakdown, runtime, and fallback count saved.
