---
round: 1
status: completed
goal: Reduce forecast error on sales while keeping runtime and holdout stable.
metric: wmape
baseline: ''
n_agents: 3
guardrails:
- runtime_s<=600
timeout_s: 1200
budget_s: 3600
---

# Research Plan - Round 1

Edit this file freely: reword hypotheses, delete or add experiments, change
the frontmatter. The edited file is exactly what runs when you type `execute`.

## Director's analysis

- explore_training_data(analysis=profile)
- analyze_errors(attempt_id=baseline)
- explore_training_data(analysis=seasonality)
- explore_training_data(analysis=intermittency)
- worst_items(attempt_id=baseline, limit=15)
- explore_training_data(analysis=drivers)

## Experiment 1: Global LightGBM panel model with leakage-safe lag/rolling features and direct 7-day stacking

- category: model
- skills: boosting-demand-models, temporal-validation-and-leakage

### Hypothesis

The baseline (WMAPE 0.4399, bias -13.3%) under-forecasts heavily, especially on Thu/Fri (bias ~-25%) and on high-volume items (12 of the 15 worst items are under-forecasts). Demand is smooth (zero share 0%, 9965/11397 items 'smooth') with strong weekly seasonality (lag-7 autocorr 0.61, Sunday lift 1.22x, Monday 0.90x). A single global LightGBM model with per-item lag-7/14/21, rolling means, weekday calendar features, and categorical sku/vertical identity will capture weekly structure and per-item levels far better than the incumbent, cutting both the negative bias and the weekday-skewed error. With only 25 days of history (~194k rows), a direct stacked model with horizon h as a feature trains in well under 600s.

### Instructions

Rewrite solution/train.py to: (1) Load training parquet; sort by sku_name/date. (2) Build features per row using only data up to that row's forecast origin: for each item, lags of sales at 7, 14, 21 days; rolling means of sales over 7 and 14 days computed on shift(1) series; expanding item mean shifted by 1; days since item first observed. (3) Calendar features: day-of-week (categorical), is_weekend, day-of-month. (4) Known/static covariates available per row: vertical (categorical), pe_flag, oos_flag, normal_day_flag, selling_price, price_index_fsp_based, price_index_mrp_based, past15daysmean, last_30_days_mean_fsp — include them only as same-row values if they exist in the request file; otherwise restrict to lag/calendar/item features (check request columns at runtime and intersect). (5) Create a direct stacked training set: for each origin day t in the last ~14 days of history, rows for horizons h=1..7 with features anchored at t and target sales at t+h; include h as a numeric feature. Also include one-step rows from earlier days to bulk up training. (6) Hold out the final 7 days as a temporal validation set for early stopping; train LightGBM (objective='tweedie', tweedie_variance_power=1.3, num_leaves=63, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, early stopping 100 rounds, categorical_feature for sku_name/vertical/weekday). (7) Predict each requested (sku_name, date) with features anchored at the last observed date per item; clip predictions at >=0; fall back to item-level lag-7 seasonal naive (or vertical weekday mean for unseen items) when history is missing. (8) Write parquet with exactly sku_name, date, forecast. Verify runtime < 600s and every requested row covered exactly once.

## Experiment 2: Per-item weekday-profile seasonal naive with recency-weighted level and global bias correction

- category: statistical-baseline
- skills: retail-demand-data, temporal-validation-and-leakage

### Hypothesis

The baseline's -13.3% bias and strong weekday error pattern (Wed +8.4% over-forecast vs Thu/Fri ~-25% under-forecast) suggest it fails to model the weekly profile (Sunday 1.22x, Monday 0.90x of mean). With only 25 days of history, a robust statistical baseline that combines each item's recent level (mean of last 14 observed days) with a per-item (fallback: per-vertical, then global) weekday multiplicative profile estimated from all history should beat the incumbent cheaply and provides a strong, stable reference. A single global multiplicative bias correction factor tuned on the last-7-day rolling origin can remove systematic under-forecasting without touching the hidden holdout.

### Instructions

Implement in solution/train.py a pure pandas/numpy forecaster: (1) For each item compute level = mean of sales over the last 14 observed days (fallback: all-history mean; fallback for unseen items: vertical mean, then global mean). (2) Estimate weekday multiplicative factors: per item, factor[dow] = mean(sales on that weekday)/mean(sales), shrunk toward 1.0 with weight n_weekday_obs/(n_weekday_obs+2); if an item has <2 observations for a weekday, use the vertical-level weekday factor; else the global weekday factor. (3) Forecast for request date with weekday d: level * factor[d]. (4) Rolling-origin self-check: repeat the same procedure anchored 7 days before the end of training, compute WMAPE on the last 7 days, and derive a single scalar calibration c = sum(actual)/sum(predicted) from that origin; apply c to all forecasts (clip to [0.5, 1.5] for safety). (5) Clip forecasts at >=0, ensure finite, write parquet with sku_name, date, forecast covering every requested row exactly once. Keep the implementation vectorized (groupby/merge only) so runtime stays a few seconds.

## Experiment 3: Price-elasticity and event-flag feature augmentation on top of the global boosting model

- category: features
- skills: boosting-demand-models, retail-demand-data, temporal-validation-and-leakage

### Hypothesis

pe_flag rows show 0.52x lift (i.e., promo-event rows sell roughly half — likely a price-event indicator), normal_day_flag rows 1.81x, and oos_flag rows 0.79x of average sales, while price_index_fsp_based varies from 0.002 to 28 (mean ~1.0). The baseline ignores these known-at-request drivers, which likely explains part of the large per-item under-forecasts on high-volume items (worst item has WMAPE 1.05 with 10.5k actual units). Adding item-relative price features (price vs item's shifted median price, discount depth vs mean_mrp) and the event flags to the global LightGBM model should capture promo/price-driven spikes and dips, reducing error on the high-volume items that dominate WMAPE.

### Instructions

Starting from the global LightGBM implementation (idea 1) — or the current solution/train.py if it already uses boosting — add ONLY these features, all computable without future target leakage: (1) price_rel_item = selling_price / item's expanding median selling_price shifted by 1 day; (2) discount_depth = 1 - selling_price/mean_mrp; (3) price_change_1d = selling_price / lag-1 selling_price - 1 (per item, shifted so no future info); (4) price_index_fsp_based and price_index_mrp_based as-is; (5) pe_flag, oos_flag, normal_day_flag, cap_bind_flag as categorical/binary features; (6) interaction pe_flag_x_discount = pe_flag * discount_depth. Guard: verify each of these columns exists in AUTORESEARCH_REQUEST for all future dates; drop any that are absent rather than fabricating values. Re-run the same temporal last-7-day validation and early stopping as the base model; keep tweedie objective and all other settings identical so this is a single-change experiment. Clip forecasts >=0, keep the seasonal-naive fallback for unseen items, and confirm runtime < 600s.
