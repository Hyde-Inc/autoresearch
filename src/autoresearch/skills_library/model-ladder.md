---
name: model-ladder
description: Choose the next model family based on baseline strength, data volume, horizon, and runtime budget.
---
Start with the cheapest model that can express the visible signal. Compare against seasonal naive before adding complexity.

- Use ETS or Holt-Winters for stable level, trend, and seasonality with limited history.
- Use SARIMA when autocorrelation is strong and the SKU count is small enough for per-series fitting.
- Use global tree models when many related SKUs share calendar, lag, price, or promotion effects.
- Use Chronos when long context, cross-series transfer, or nonlinear patterns justify its cost.
- Escalate only when the previous model's residuals show structure the next family can capture.
- Keep a fallback for sparse or short-history SKUs.
