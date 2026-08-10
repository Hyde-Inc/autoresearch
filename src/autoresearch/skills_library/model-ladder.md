---
name: model-ladder
description: Route a demand task to seasonal/statistical, intermittent, boosting, or foundation models using data size, forecast horizon, known-future drivers, residual structure, dependency availability, and runtime budget.
---
# Demand model-family routing

This skill chooses the next experiment family; pair it with the selected
family's implementation skill. Prefer the cheapest model that can express a
specific residual pattern. "Try a more powerful model" is not a hypothesis.

## Profile before selecting

Ground the decision in:

- number of items, rows per item, frequency, and forecast horizon;
- seasonal strength at relevant periods;
- zero share, ADI/CV², and short-history item count;
- trend/regime changes and age of items;
- known-future calendar, price, promotion, holiday, and availability fields;
- incumbent residuals by item, horizon, weekday, and demand scale;
- installed libraries, cold-start cost, CPU/GPU, and runtime guardrail.

## Routing table

**Seasonal naive / recent mean**

Use as a required baseline and fallback. It is often strong when weekly
seasonality dominates and the horizon is short. If a proposed model cannot beat
it on a matching temporal origin, fix validation or abandon the complexity.

**ETS / Holt-Winters**

Use for dense per-item series with stable level/trend/one dominant seasonality,
limited covariates, and enough cycles. It is cheap, interpretable, and often
better than ARIMA on smooth demand.

**ARIMA / SARIMA / SARIMAX**

Use when residual autocorrelation remains after baseline seasonality and item
count is small enough for per-series fits. SARIMAX is justified only when
future exogenous values are known. Avoid expensive order searches over large
panels; route eligible items and retain fallbacks.

**Croston / SBA / TSB / hurdle**

Use for items with long zero intervals. SBA targets stable intermittency; TSB
handles changing occurrence probability; hurdle models use shared known-future
drivers. Do not apply intermittent methods to dense items.

**Global XGBoost / LightGBM / CatBoost**

Use when many related items share lag, calendar, category, price, or promotion
effects, especially with short per-item histories. Requires explicit
forecast-origin feature construction and temporal validation. Prefer direct
multi-horizon training for a fixed multi-step request.

**Chronos / Chronos-Bolt / Chronos-2**

Use as a measured zero-shot prior when the package/checkpoint is available,
histories contain useful context, and load/inference fit the budget. Bolt is a
univariate direct-quantile model; Chronos-2 is the path when current package
support and known-future covariates justify it. Route sparse/short items away.

**Ensemble**

Use only after two independently valid models make complementary
out-of-sample errors. Blending two versions of the same feature/model pipeline
rarely justifies doubled runtime.

## Experiment sequencing

1. Verify incumbent and seasonal-naive scores on the exact evaluator.
2. Fix obvious data/coverage/bias defects before changing model family.
3. Add one family whose inductive bias matches measured residual structure.
4. Keep all other routing/fallback behavior unchanged so the result is
   attributable.
5. If it wins validation, verify on another rolling origin when available.
6. Ensemble only after standalone candidates are stable.

Every proposal should state:

- **evidence**: the measured property that motivates the family;
- **change**: one model/routing/feature change;
- **expected effect**: which item/horizon/error segment should improve;
- **falsifier**: what result would reject the hypothesis;
- **cost**: dependency, checkpoint, fit count, and expected runtime;
- **fallback**: behavior for unsupported or failed items.

## Avoid false sophistication

- Do not propose Chronos because it is a foundation model, boosting because it
  wins competitions, or ARIMA because the data is a time series.
- Do not run several model families inside one experiment; parallel experiments
  exist to compare them cleanly.
- Do not tune against hidden holdout results.
- Do not add a dependency without checking the project environment and budget.
- Do not use a model whose cold-start/download dominates a live demo.
- Do not replace a working dense-item model while testing sparse-item routing.

## Minimum acceptance check

A family is eligible only if it can produce every requested item/date,
stay finite and non-negative, fit the training timeout, and preserve a simple
fallback. Accuracy gains that violate those conditions are not valid wins.
