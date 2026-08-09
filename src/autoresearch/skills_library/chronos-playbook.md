---
name: chronos-playbook
description: Decide when and how to use Chronos/Chronos-Bolt for demand forecasting - the zero-shot-first ladder, context and horizon limits, covariate regressors, fine-tuning recipes, and the situations where Chronos is the wrong tool.
---
Chronos is a pretrained time-series foundation model. Use it as a strong prior,
not a default: measure zero-shot first, and only add complexity that beats it
on the protected validation split.

The ladder - stop at the first rung that wins:

1. Zero-shot Chronos-Bolt. Prefer Bolt variants over original Chronos: they
   forecast in one direct pass (no autoregressive sampling), output quantiles
   directly, run fine on CPU, and are ~250x faster - which matters under a
   runtime_s guardrail. Start with the smallest checkpoint (mini or small) and
   only move to base if the smaller one is clearly underfitting.
2. Zero-shot plus a covariate regressor (see below) when promotions or price
   changes visibly move demand.
3. Fine-tuning, only after 1 and 2 are measured, and only with many related
   series available.

Limits to respect in every situation:

- Context truncates at 2048 steps; feeding longer history is silently wasted.
  Match context to the longest useful seasonal cycle (for daily retail data,
  365-730 days captures yearly seasonality).
- Quality degrades beyond the native 64-step prediction length: autoregressive
  rollout collapses the variance of the forecast. A 28-day daily horizon fits
  in one direct pass; never chain rollouts for it.
- Use the median (0.5) quantile for symmetric point metrics like wmape; pick a
  higher quantile only when the metric or guardrail explicitly penalizes
  under-forecasting more than over-forecasting.
- Scale each series (e.g. mean or standard scaler per SKU) and restore units
  before writing forecasts; unscaled heavy-tailed series hurt the tokenizer.

Covariates: Chronos-Bolt is univariate. To use promo/price/calendar signals,
wrap it with a covariate regressor: a small tabular model (e.g. ridge or
LightGBM) fits the covariate effect, Chronos forecasts the residual/adjusted
target, and the effect is added back. Always A/B this against pure zero-shot -
covariate regressors sometimes hurt when the signals are weak or noisy.

Fine-tuning recipe (rung 3):

- Train at the task's prediction length and the same context length used at
  inference; mismatches quietly cost accuracy.
- LoRA with learning rate around 1e-5, or full fine-tune around 1e-6; ~1000
  steps is usually enough. Early-stop on a validation window that the training
  windows never overlap.
- Needs many related series (dozens+); fine-tuning on a handful of series
  memorizes the validation period and collapses on the holdout.

When NOT to use Chronos:

- Histories shorter than roughly two seasonal cycles: fall back to seasonal
  naive or simple exponential smoothing.
- Heavily intermittent SKUs (most days zero): route those to Croston/SBA or a
  Tweedie tree model and let Chronos handle the smooth movers - a routed
  hybrid usually beats either model applied to everything.
- When a tuned tree model already wins on validation: then blend instead of
  replacing - equal-weight or error-weighted averaging of Chronos and tree
  forecasts helps precisely when their errors are complementary (check the
  per-SKU error correlation before bothering).
