---
name: ensembling
description: Blend complementary statistical, tree, and foundation-model forecasts when their errors differ.
---
Ensemble only models that make meaningfully different errors.

- Start with a simple average or validation-derived nonnegative weights that sum to one.
- Compare residual correlation by model and SKU; highly correlated models add little diversity.
- Blend toward seasonal naive when a complex model is unstable across time or sparse SKUs.
- Fit weights on out-of-sample validation predictions, never in-sample fitted values.
- Prefer one global weight unless there is enough evidence for segment-specific weights; shrink local weights toward global.
- Keep runtime within budget by reusing forecasts and avoiding redundant model fits.
