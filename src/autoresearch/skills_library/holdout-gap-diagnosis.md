---
name: holdout-gap-diagnosis
description: Diagnose experiments that improve validation but regress on the later hidden holdout period.
---
A validation win with a holdout loss usually signals temporal overfitting, regime change, or unstable tuning.

- Compare validation and holdout bias, scale, trend, zero rate, price, and promotion frequency.
- Remove parameters tuned narrowly to one validation window; prefer robust defaults or multiple rolling origins.
- Shorten stale history or weight recent observations more when the level changed.
- Reduce feature/model complexity when residual gains concentrate in a few dates or SKUs.
- Test calibration and ensembling against seasonal naive because a conservative blend often survives drift better.
- Do not use holdout values for fitting or parameter selection. Use only the pattern of the failure to form the next hypothesis.
