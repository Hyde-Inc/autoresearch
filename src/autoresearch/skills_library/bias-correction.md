---
name: bias-correction
description: Correct systematic over-forecast or under-forecast bias without hiding segment failures.
---
Measure signed error overall and by SKU, demand scale, and horizon.

- Apply multiplicative calibration when error scales with demand and additive calibration when it is roughly constant.
- Estimate correction from out-of-sample residuals, not training fit residuals.
- Shrink per-SKU corrections toward a global correction when SKU history is short.
- Clip only to physically valid bounds and inspect whether clipping creates new under-forecast bias.
- If business costs are asymmetric, tune the target quantile or loss rather than applying an arbitrary final multiplier.
- Preserve WMAPE and RMSE guardrails because a zero aggregate bias can hide large offsetting errors.
