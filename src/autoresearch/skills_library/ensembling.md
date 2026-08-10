---
name: ensembling
description: Implement safe forecast blends and bias calibration from temporal out-of-sample predictions, with constrained weights, segment shrinkage, residual-correlation checks, and no hidden-holdout tuning.
---
# Ensembling and calibration

Ensemble only independently valid candidates whose temporal out-of-sample
errors are complementary. A blend is not a way to rescue a broken component
or hide leakage.

## Preconditions

Every component must:

- forecast the exact same item/date rows from the same origin;
- be finite, non-negative, and independently backtested;
- preserve its own fallback;
- fit the combined runtime/dependency budget.

Join by keys and assert one-to-one coverage before combining. Never rely on row
order.

## Start with fixed convex blends

For two forecasts:

```python
blend = weight * forecast_a + (1.0 - weight) * forecast_b
```

Test a tiny predeclared grid such as `{0.25, 0.5, 0.75}` on validation or
rolling-origin predictions. Weights must be non-negative and sum to one; this
keeps the blend within component bounds and reduces overfit. Equal weight is
the required baseline.

Use residual correlation to decide whether tuning is worthwhile:

```python
residuals = predictions.assign(
    err_a=lambda x: x.forecast_a - x.actual,
    err_b=lambda x: x.forecast_b - x.actual,
)
correlation = residuals[["err_a", "err_b"]].corr().iloc[0, 1]
```

High correlation and similar bias mean little ensemble value. Useful pairs
often combine distinct inductive biases: seasonal naive + boosting,
statistical + Chronos, or dense-item model + intermittent router.

## Fit weights only with enough out-of-sample data

When several rolling origins exist, constrained least squares is reasonable:

```python
from scipy.optimize import nnls

matrix = np.column_stack([pred_a, pred_b, pred_c])
weights, _ = nnls(matrix, actual)
weights = weights / weights.sum() if weights.sum() else np.full(3, 1 / 3)
```

Do not add SciPy just for a two-model blend; use a fixed grid. Fit weights on
out-of-sample forecasts, never training fitted values and never the hidden
holdout. Report each component score alongside the ensemble.

## Segment-specific weights require shrinkage

One global weight is safest. Use segment weights only when a stable,
training-derived segment (for example sparse vs dense) has enough validation
mass across multiple origins. Shrink a local estimate toward global:

```python
shrunk = (n * local_weight + strength * global_weight) / (n + strength)
```

Avoid per-SKU weights on one 28-day window; they memorize noise. Horizon-
specific weights have the same risk unless multiple origins support them.

## Bias calibration

Calibrate only a stable, systematic out-of-sample bias:

- multiplicative factor when error scales with demand;
- additive shift when error is roughly constant;
- target quantile when business loss is asymmetric.

```python
factor = actual.sum() / max(forecast.sum(), 1e-12)
factor = np.clip(factor, 0.8, 1.2)  # predeclared safety bound
calibrated = np.clip(forecast * factor, 0, None)
```

Estimate from rolling-origin or validation predictions, not in-sample fitted
values. Check that calibration improves the primary metric as well as bias;
zero aggregate bias can hide large offsetting item errors. Shrink per-segment
factors toward 1 and never calibrate each item from a tiny window.

Log transforms often create negative bias after inversion; Tweedie/quantile
models can create positive bias. Fix the model/objective first when possible,
then apply a bounded calibration as a final measured change.

## Runtime-safe implementation

Reuse already-generated component forecasts when the evaluation architecture
allows it. Otherwise account for both fits and checkpoint loads in runtime.
A marginal metric gain is invalid if the combined model violates the training
guardrail or depends on an unavailable model artifact.

## Required checks before reporting

- exact one-to-one key alignment across actuals/components;
- component scores, signed biases, and residual correlation;
- fixed/global weight or training-only fitting procedure stated;
- no holdout actual used for weights, segments, or calibration;
- ensemble beats equal-weight and the best component, not merely the worst;
- all outputs remain finite/non-negative after calibration;
- combined cold/warm runtime and fallback behavior measured.
