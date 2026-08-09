---
name: xgboost-demand
description: Configure gradient-boosted trees for demand forecasting - choosing the objective for zero-heavy demand, direct vs recursive multi-step strategy, the features that pay, training hygiene, and stockout bias traps (M5-style practice).
---
Tree models win demand competitions when the objective, the multi-step
strategy, and the validation split match the task. Get those three right
before tuning anything else.

Objective - condition on how zero-heavy the demand is:

- Zero-heavy or intermittent demand: use `reg:tweedie`. Tune
  `tweedie_variance_power` in 1.1-1.5 - toward 1.1 for mildly intermittent,
  toward 1.5 for heavy intermittency. Plain squared error systematically
  under-forecasts intermittent SKUs because it averages over the zeros.
- Smooth, always-selling series: squared error on a log1p-transformed target
  is fine; invert with expm1 and clip at zero before writing forecasts.
- Never use log1p and Tweedie together; Tweedie expects the raw scale.

Multi-step strategy for a 28-day-style horizon:

- Direct per-horizon models (one model per step, or one model with a horizon
  feature) avoid error compounding and are the safer default.
- If recursive, never feed lag_1: the model over-relies on it and compounds
  its own errors step by step. Use lags at or beyond the horizon (28, 35, 42
  for a 28-day horizon) directly, and prefer rolling means (7/14/28-day) over
  short lags to smooth the signal.
- When the budget allows, ensemble a recursive and a direct variant with
  equal weights; their error patterns differ and the blend is robust.

Features that pay (in rough order of value):

- Rolling mean/std of demand over 7/14/28 days, shifted so nothing from the
  forecast window leaks in.
- Price: change vs yesterday and price normalized by the SKU's own average -
  relative price moves matter more than absolute levels.
- Promo flags with lead and lag versions (demand rises before and dips after
  promotions), and day-of-week x promo interactions.
- Calendar: day-of-week, week-of-year, month; item age / weeks since release
  for newer SKUs.
- Expanding shifted mean-encodings of SKU and category. Never plain (unshifted)
  mean encoding - it leaks the target.

Training hygiene:

- Validate on a temporal split whose length matches the real forecast horizon;
  random splits overstate accuracy badly.
- Early stopping on that split; moderate depth (6-10) with min_child_weight
  raised to fight overfit; subsample and colsample_bytree around 0.5-0.8.
- One global model pooled across all SKUs beats per-SKU models when history
  per SKU is short; encode SKU as a categorical or stable integer ID.

Bias traps:

- Long zero runs may be stockouts, not zero demand. Training on them teaches
  the model to under-forecast; treat suspicious runs (many consecutive zeros
  for an otherwise-selling SKU) as missing rather than zero.
- After any objective or transform change, check the bias_pct guardrail: trees
  calibrated on Tweedie can drift positive, log1p models drift negative. A
  final multiplicative calibration on the validation window is cheap and safe.
