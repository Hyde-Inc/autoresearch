---
name: chronos-playbook
description: Implement and debug Amazon Chronos zero-shot demand forecasts with the current chronos-forecasting APIs, including Chronos-Bolt batching, Chronos-2 covariates, checkpoint/cache preflight, quantile extraction, routing, and runtime-safe fallbacks.
---
# Chronos foundation models for demand

Chronos is an optional pretrained-model experiment, not a guaranteed upgrade.
Start zero-shot, use the smallest adequate checkpoint, and compare against
seasonal naive on exactly the protected forecast horizon.

## Preflight before writing model code

Chronos failures are commonly environment/download failures, not forecasting
results. Check these first and stop cleanly if they fail:

1. `chronos-forecasting` is installed and imports in the project environment.
2. PyTorch supports the project Python version and selected device.
3. The requested Hugging Face checkpoint is already cached or network access is
   available. A model download can exceed an agent session; never poll it from
   the coding agent.
4. Available memory fits the checkpoint and item batch.

```python
import importlib.metadata as md
import torch
from chronos import BaseChronosPipeline

print(md.version("chronos-forecasting"), torch.__version__)
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

Pin the dependency/checkpoint only when the experiment is allowed to edit the
environment. Otherwise report a missing dependency or unavailable checkpoint;
never silently substitute another model under a Chronos experiment name.

## Chronos-Bolt: practical univariate zero-shot path

The current public package exposes original Chronos and Chronos-Bolt through
`BaseChronosPipeline`. Bolt directly predicts quantiles and is the practical
CPU choice:

```python
import torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-mini",
    device_map="cpu",
    torch_dtype=torch.float32,
)

# One 1-D tensor per item; variable lengths are supported.
contexts = [
    torch.tensor(group["units_sold"].to_numpy(), dtype=torch.float32)
    for _, group in history.sort_values("date").groupby("sku_id", sort=False)
]
quantiles, mean = pipeline.predict_quantiles(
    contexts,
    prediction_length=horizon,
    quantile_levels=[0.1, 0.5, 0.9],
)
# quantiles shape: [num_series, horizon, 3]
point = quantiles[:, :, 1].numpy()
```

API details matter:

- `predict_quantiles` returns `(quantiles, mean)`. Quantiles have shape
  `[batch, prediction_length, num_quantile_levels]`.
- `predict` on Chronos-Bolt returns its trained quantile grid with shape
  `[batch, num_quantiles, prediction_length]`; do not mistake axis 1 for time.
- Pass a 1-D tensor, a list of 1-D tensors, or a left-padded 2-D tensor.
- Chronos performs its own per-series scaling. Do not manually standardize and
  then forget to invert; custom scaling is an experiment, not a requirement.
- Preserve the item order explicitly when converting tensors back to rows.

Start with `bolt-tiny` or `bolt-mini` for CPU demos and move to `bolt-small`
only when measured quality justifies runtime. Load the pipeline once per
`train.py`, then batch items; loading once per SKU is a severe bug.

## Context and horizon

- Official Chronos-Bolt has a default context limit of 2048 observations; use
  the most recent useful context when history is longer.
- Bolt's native prediction length is 64. Longer requests are produced by
  repeated chunks and can suffer variance collapse at chunk boundaries. A
  28-day horizon fits in one pass.
- Keep enough history for at least two relevant seasonal cycles. Very short
  items should route to a baseline.
- Ensure each item is regular at the task frequency and decide whether missing
  dates mean zero demand or missing observations before tensor creation.

For WMAPE/RMSE point forecasts, begin with the median (`0.5`) quantile. Use a
higher quantile only when the configured business loss penalizes shortages
more heavily. Always clip final demand forecasts at zero and verify finiteness.

## Chronos-2 when known-future covariates matter

Chronos-Bolt is univariate. Do not claim it consumed promo/price columns.
Current `chronos-forecasting` releases also expose `Chronos2Pipeline`, whose
dataframe API can consume covariates:

```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)
pred = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=horizon,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="sku_id",
    timestamp_column="date",
    target="units_sold",
)
```

Check the installed package's `predict_df` signature because this API evolves.
Only pass future price/promotion values that are truly known for every request
date. If Chronos-2 is unavailable, either use pure Bolt or model a covariate
effect separately and A/B the combined result; do not invent future drivers.

## Routing and fallback

Chronos often struggles on:

- mostly-zero intermittent series;
- histories shorter than two seasonal cycles;
- structural breaks with no relevant context;
- tiny panels where checkpoint load dominates runtime.

Route these to seasonal naive, ETS, or Croston/SBA. Build a `can_use_chronos`
mask from training history only and preserve a baseline forecast for every
item. If inference raises, returns a wrong shape, or produces non-finite values,
replace only the affected series with its fallback.

## Fine-tuning is a separate research project

Do not fine-tune in the first Chronos attempt. It requires many related series,
window generation that never overlaps validation, checkpoint persistence, and
enough compute to fit inside the runtime contract. On a handful of demand
series, zero-shot plus routing or an ensemble is usually more defensible.

## Required checks before reporting

- package version, device, checkpoint, download/cache status, and load time;
- context lengths and number of routed/fallback series;
- output tensor axes and item/date reconstruction verified on a tiny batch;
- one pipeline load and batched inference;
- median point forecast compared with seasonal naive on the same origin;
- finite, non-negative, exact-coverage output;
- runtime split into checkpoint load and inference so demo cold-start cost is
  visible.
