---
name: feature-selection-tree-models
description: "Add and remove features to XGBoost/LightGBM/CatBoost demand models in named blocks (calendar, payday, pricing, lag/rolling, promo) rather than one at a time, and keep or drop the whole block based on directional validation improvement. Use whenever iterating on the feature set of a tree-based forecaster, not only for quick-commerce — this applies to any panel/global boosting model."
---
# Block-wise feature selection for tree models

Testing one feature at a time on a boosting/tree model is slow and often
uninformative for demand forecasting, because a single feature rarely moves a
tree ensemble's error on its own. Trees find interactions, so a feature's
contribution usually only shows up combined with a few related features.
Group related features into a named block and add or remove the whole block
at once; a genuinely useful block shows a directional improvement immediately,
without needing to isolate individual features inside it first.

## Define blocks before experimenting

- Group by shared signal, not by data source: a *calendar* block (day-of-week,
  day-of-month, month, holiday flag), a *payday* block (payday-window flag,
  days-from-boundary), a *pricing* block (relative price, discount depth,
  promo flags), a *lag/rolling* block (lag_7/14/28, rolling means).
- Aim for blocks of roughly 3-8 related features. A block of 1 feature is just
  single-feature testing with an extra label; a block of 30 unrelated features
  makes it impossible to tell what mattered if it wins or loses.
- State a hypothesis for each block before running it — which residual pattern
  it should fix — the same discipline the `model-ladder` skill requires for
  model-family changes. A feature block is an experiment, not a free action.

## Test procedure

1. Score the current accepted feature set on the horizon-matched temporal
   backtest (see `temporal-validation-and-leakage`) as the baseline to beat.
2. Add one block, retrain, and compare validation score and signed bias
   against that baseline.
3. A directional improvement — better score, or the same score with reduced
   bias/variance on the pattern the block targets — is grounds to keep the
   whole block. A flat or worse result is grounds to drop the whole block, not
   to start removing individual features from it.
4. Only descend into per-feature trimming inside a block when the block
   already showed improvement but part of it looks like dead weight (for
   example several members with near-zero importance) and the goal is to cut
   cost or latency. This is a refinement step after the block has already
   proved useful, not the primary search strategy.
5. Keep accepted blocks in the feature set and stack the next block on top of
   the new baseline, so later blocks are tested cumulatively in the presence
   of earlier accepted ones, not in isolation.

```python
blocks = {
    "calendar": ["dow", "dom", "month", "is_holiday"],
    "payday": ["days_from_payday_boundary", "is_staple_vertical"],
    "pricing": ["rel_price", "discount_depth", "is_promo"],
}

accepted = list(base_features)
baseline_score, baseline_bias = evaluate(accepted)

for name, cols in blocks.items():
    trial_score, trial_bias = evaluate(accepted + cols)
    improved = trial_score < baseline_score or abs(trial_bias) < abs(baseline_bias)
    print(name, "keep" if improved else "drop", trial_score, trial_bias)
    if improved:
        accepted += cols
        baseline_score, baseline_bias = trial_score, trial_bias
```

## Why this beats one-by-one for trees

- Trees split on whatever feature reduces loss most at each node. A single new
  feature competes against every existing feature for split priority and can
  show near-zero importance alone while being essential in combination — for
  example `days_from_payday` alone may look useless, but its interaction with
  `vertical_elasticity`, which the tree can only discover if both exist
  together, is not.
- One-by-one testing on a noisy demand target also wastes runtime on features
  whose individual signal sits below the noise floor of a single backtest
  window. A block amortizes that noise over several correlated signals, so the
  direction of the result is more trustworthy.
- Cost still matters: track feature count and training/inference time per
  accepted block. Trees don't automatically ignore redundant or unhelpful
  columns at negligible cost, especially with many category encodings.

## Guardrails

- Don't build blocks so large or so unrelated that a pass/fail verdict becomes
  meaningless — that's re-running the whole feature search inside one commit.
- Don't tune the acceptance threshold against the hidden holdout; use the same
  temporal validation/backtest procedure as any other experiment.
- Re-run leakage checks per block, not only once at the start. Payday and
  pricing blocks are exactly the kind of feature group prone to accidentally
  encoding future information (see `retail-demand-data` and
  `temporal-validation-and-leakage`).

## Required checks before reporting

- feature blocks named and defined before any experiment runs, each with a
  stated hypothesis;
- baseline score/bias recorded before the first block is tested;
- keep/drop decisions made at the block level, with per-feature trimming only
  as a secondary refinement on an already-accepted block;
- blocks tested cumulatively against the growing accepted feature set, not
  only against the original baseline;
- leakage/known-future checks re-run for each new block;
- feature count and training/inference cost tracked per accepted block.
