---
name: retail-demand-data
description: "Audit and model retail demand data semantics across model families: missing dates, stockout-censored sales, returns, planned price/promotion covariates, holidays, cold starts, cannibalization, and hierarchy reconciliation."
---
# Retail demand data and covariates

Use this skill before attributing forecast error to a model. Observed sales are
not always latent demand, and a feature that exists historically is not
automatically available for future request dates.

## Establish the row contract

For each item/location/date, identify:

- target meaning: ordered, shipped, invoiced, or sold units;
- frequency/time zone and whether missing dates mean zero, closed, or absent;
- duplicate aggregation rule;
- treatment of returns/cancellations and negative values;
- inventory/availability and stockout flags;
- item lifecycle, launch/discontinuation, and assortment changes;
- whether future price, promo, holiday, and availability are planned inputs.

Write assertions for uniqueness, monotonic dates per series, frequency, target
bounds, and exact request coverage. Do not silently aggregate stores or SKUs
unless the task contract permits it.

## Missing rows are not automatically zero demand

Reindex each series to the task calendar, then classify gaps:

- store open + item ranged + observed no sale: likely zero;
- store closed / item not ranged: structural missing, not zero;
- pipeline outage: missing observation;
- stockout: sales are censored below latent demand;
- before launch / after discontinuation: outside lifecycle.

Using `.asfreq("D").fillna(0)` is valid only after that classification.
Unexplained zero filling can manufacture intermittency and teach every model to
under-forecast.

## Stockout-censored sales

Long zeros or low sales while on-hand inventory is zero are not evidence of
zero demand. Depending on available fields:

- exclude censored target rows from supervised loss while retaining calendar
  continuity;
- impute cautiously from same-weekday/category demand;
- add availability as a known-future covariate when replenishment plans provide
  it;
- evaluate separately on available periods.

Never "recover" demand using validation/holdout sales. State whether the task
scores observed sales or latent demand: optimizing one does not guarantee the
other.

## Price and promotions

Separate planned fields from realized fields. For every future covariate, assert
that the request contains a value for every forecast date; otherwise define a
fallback scenario before training.

Useful leakage-safe features:

- price relative to the item's shifted historical median;
- percentage/absolute change from the last known or planned price;
- discount depth relative to regular price;
- promo type, duration, days since start, days to end;
- calendar/promo interactions and category-level promo intensity.

Promo and price are often endogenous: weak items may be discounted, and promo
plans target expected demand. Treat feature importance as predictive, not
causal elasticity. Monotonic price constraints can stabilize a boosting model
when higher price should not increase demand, but exceptions and promotion
bundles must be checked first.

Use training-only ablations:

1. baseline without price/promo;
2. planned future fields only;
3. add derived interactions.

Report promo and non-promo errors separately. A model that wins only on a few
promo spikes may regress ordinary replenishment.

## Calendar, events, and multiple seasonalities

Calendar values are deterministic known future features:

- weekday/weekend, week/month/quarter;
- holidays and distance to holiday;
- payday, school term, local event when available;
- days since launch and seasonal age.

Use locale/store-appropriate calendars. Week-of-year has year-boundary issues;
prefer cyclic encodings for linear models or ordinary integers/categories for
trees. A short dataset cannot identify annual seasonality merely because
`month` is present.

## Cold start and item lifecycle

For unseen or short-history items, target lags do not exist. Provide a route:

- category/store seasonal profile scaled by known item attributes;
- pooled global boosting model using static and known-future features;
- analog item selected from training metadata;
- conservative category mean/seasonal-naive fallback once enough history
  arrives.

Do not encode unseen IDs as an arbitrary existing category. Preserve a known
unknown code and test it on deliberately held-out items. Lifecycle features
must be based on dates known at origin.

## Substitution and cannibalization

Price/promo changes on one item can move demand to substitutes. When assortment
relationships are available, add category-level aggregate demand, competitor
promo counts, or relative price rank—but compute target-derived aggregates
through the forecast origin only. Without relationship data, broad cross-item
lags can add leakage and noise; test them as a separate experiment.

## Hierarchy and reconciliation

Item forecasts may need to sum to category/store totals. First verify whether
the evaluator scores bottom-level rows only. If coherence matters:

- bottom-up: sum item forecasts (simple, preserves item model);
- top-down: allocate an aggregate forecast using training-only historical
  shares (weak for changing assortment);
- reconciliation: adjust forecasts using out-of-sample error covariance when
  enough origins exist.

Do not add reconciliation to a bottom-level metric without measuring it; a
coherent hierarchy can have worse item WMAPE.

## Required checks before reporting

- explicit meaning for missing, zero, negative, stockout, prelaunch, and
  discontinued rows;
- every future covariate proven available for every request date;
- train-only fitting for medians, category shares, encoders, and imputers;
- promo/non-promo, available/stockout, and new/existing-item diagnostics;
- unknown-category and short-history fallback tested;
- no causal claims from predictive feature importance;
- exact item/date output and physically valid forecasts.
