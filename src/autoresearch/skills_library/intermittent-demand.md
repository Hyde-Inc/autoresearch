---
name: intermittent-demand
description: Implement and route intermittent-demand forecasts with Croston, SBA, TSB, ADIDA, or hurdle models, including segment diagnostics, equations, edge cases, and evaluation under zero-heavy demand.
---
# Intermittent and lumpy demand

Use this skill when demand has long zero runs and irregular nonzero sizes.
Intermittency is not simply "low volume": measure occurrence and positive-size
processes separately and route only the affected items.

## Diagnose from training history

For each item compute:

- zero share and number of nonzero observations;
- average demand interval `ADI`: periods / nonzero demands;
- squared coefficient of variation of positive sizes `CV²`;
- recent vs long-run occurrence probability;
- longest zero run and time since last nonzero demand.

The common Syntetos-Boylan segmentation uses `ADI=1.32` and `CV²=0.49` as
rough boundaries:

- smooth: low ADI, low CV²;
- erratic: low ADI, high CV²;
- intermittent: high ADI, low CV²;
- lumpy: high ADI, high CV².

Treat these as routing priors, not universal truth. Require enough nonzero
events (for example 5-10) before trusting per-item parameter estimates.

## Croston and SBA

Croston updates positive size `z` and inter-arrival interval `p` only when a
nonzero demand occurs, then forecasts `z / p`. SBA applies the approximate
bias correction `(1 - alpha / 2) * z / p`.

```python
import numpy as np

def croston_sba(y, horizon, alpha=0.1):
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(horizon)

    z = y[nz[0]]
    p = float(nz[0] + 1)
    last = nz[0]
    for index in nz[1:]:
        interval = index - last
        z = alpha * y[index] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
        last = index

    level = (1 - alpha / 2) * z / max(p, 1e-12)
    return np.full(horizon, max(0.0, level))
```

Verify initialization and interval convention against a hand-worked sequence;
Croston implementations differ there. Tune `alpha` on rolling origins from a
small explicit set such as `{0.05, 0.1, 0.2, 0.3}`. Do not optimize it on the
hidden holdout.

Croston/SBA returns an expected demand rate per period, not a schedule of
individual sale events. Fractional forecasts are valid unless the task
contract explicitly requires integer units.

## TSB for obsolescence or changing occurrence

Teunter-Syntetos-Babai separately smooths:

- probability of nonzero demand every period; and
- positive size only on nonzero periods.

Forecast is `probability * size`. Unlike Croston, occurrence probability
decays through a long zero run, so TSB is preferable when items can become
obsolete or availability changes. Use separate smoothing parameters for
probability and size only when rolling-origin evidence supports the extra
degree of freedom.

## Hurdle model for known drivers

When occurrence depends on weekday, promotion, item category, or price:

1. fit a classifier for `P(y > 0 | x)`;
2. fit a positive-demand regressor on rows where `y > 0`;
3. multiply probability by expected positive size.

Build both models globally when per-item histories are short. Use only
known-future covariates and leakage-safe lag features. Calibrate the classifier
on temporal out-of-sample predictions; an overconfident probability model
causes direct forecast bias.

Tree backends can implement the two stages, or use a simple logistic/ridge
pair when dependencies are limited. Compare the hurdle expectation against
SBA and a recent nonzero-rate baseline before tuning.

## Temporal aggregation

ADIDA/IMAPA can stabilize very sparse daily series:

1. aggregate demand into non-overlapping weekly or multi-day buckets;
2. forecast aggregated demand with a simple model;
3. divide/disaggregate back to daily expected demand.

Only use aggregation aligned with the request calendar. Uniform
disaggregation loses weekday structure; distribute using training-only
weekday shares when there are enough nonzero events.

## Routing

Keep the dense-item incumbent untouched when the experiment is specifically
testing sparse-item handling:

```python
zero_share = history.groupby("sku_id")["units_sold"].apply(lambda s: (s == 0).mean())
sparse_ids = zero_share[zero_share >= 0.5].index
```

The threshold must come from training data and be stated explicitly. A robust
router typically uses:

- all-zero: zero or business floor;
- too few nonzeros: pooled category rate or simple recent mean;
- intermittent stable occurrence: SBA;
- declining occurrence: TSB;
- driver-dependent occurrence with enough panel data: hurdle;
- dense: incumbent statistical or boosting model.

## Evaluation traps

- Ordinary MAPE is undefined or misleading on zero actuals. Use the configured
  protected metric and inspect WMAPE, MAE, signed bias, and error on nonzero
  days.
- A constant expected-rate forecast can score well on aggregate WMAPE while
  missing every occurrence. If service timing matters, also inspect occurrence
  precision/recall or inventory simulation—but do not replace the official
  promotion metric.
- Do not round. Rounding small expected rates to zero creates systematic
  under-forecasting.
- Distinguish true zero demand from stockout-censored sales. Long zero runs for
  formerly dense items may be missing availability, not intermittency.

## Required checks before reporting

- segment counts, thresholds, nonzero-event counts, and route per item;
- hand-check of Croston/SBA initialization on a short sequence;
- finite, non-negative forecast for all-zero and one-nonzero edge cases;
- same routing rule applied to validation and production request;
- sparse and dense metric/bias reported separately;
- unchanged dense-item fallback so the experiment isolates sparse handling.
