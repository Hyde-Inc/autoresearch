---
name: cannibalization-effects
description: "Detect and correct for cannibalization when near-identical SKUs (same use-case, price band, pack size) go out of stock and shift demand onto a sibling SKU rather than losing it outright. Use top-down vertical/cluster-level forecasting with a clean (both-in-stock) share to disaggregate, so ARIMA, lag, and rolling-mean models aren't corrupted by demand-shift noise mistaken for organic growth or organic loss. Trigger whenever two or more SKUs are close substitutes and either has stockout history."
---
# Cannibalization from near-identical SKU stockouts

When two or more SKUs are close substitutes — same use case, similar price,
same pack size — one going out of stock does not remove that demand. Most of
it reappears on the sibling SKU within a few days. Item-level models (ARIMA,
lag features, trailing-X-day averages) read that reappearance as organic
growth on the substitute and organic loss on the out-of-stock item. Both
readings are wrong, and training on them corrupts the model long after the
stockout ends.

## Identify substitute clusters and confirm the signature

- Group SKUs by vertical/sub-category, price band, pack size, and use case.
  Flavor or minor variant differences alone usually don't break
  substitutability; price band and pack size do.
- Confirm substitutability from data before trusting the grouping: when item
  A's availability drops to zero, does item B show a level shift up within a
  short window (1-3 days), and does the *combined* A+B volume stay roughly
  flat rather than collapsing? That combined-flat-but-individually-shifted
  pattern is the signature of cannibalization — distinct from a real demand
  shock (both move together) or a genuine demand loss (combined total drops).

```python
oos_days = availability.loc[availability["sku_id"] == sku_a, "in_stock"].eq(False)
cluster = frame[frame["sku_id"].isin([sku_a, sku_b])]
combined = cluster.groupby("date")["units_sold"].sum()
combined_baseline = combined[~combined.index.isin(oos_days[oos_days].index)].mean()
combined_during_oos = combined[combined.index.isin(oos_days[oos_days].index)].mean()
shift_ratio = combined_during_oos / combined_baseline  # near 1.0 supports cannibalization, not loss
```

## Why item-level models break

- The substitute SKU's history shows an abrupt level jump exactly aligned with
  the sibling's stockout. ARIMA/lag/rolling-mean models read this as a
  legitimate regime change and keep the fitted level inflated even after both
  SKUs are back in stock.
- Symmetrically, the out-of-stock item's rolling means and short lags crash
  toward zero during the stockout, so once it returns to stock the model
  under-forecasts because recent history looks artificially low.
- If the two SKUs alternate stockouts, both histories become a patchwork of
  true demand and borrowed/donated demand, and the corruption compounds.

## Fix: top-down at the cluster level, trained on a clean share

- Forecast at the vertical/cluster level first (sum of all substitutes in the
  group). Cannibalization noise mostly cancels out at this level because
  total demand is far more stable than the individual split.
- Disaggregate the cluster forecast down to each SKU using a *clean share*,
  estimated only from days when every SKU in the cluster was simultaneously in
  stock. Excluding single- or multi-stockout days from the share estimate is
  the step that matters — including them lets the corrupted split leak back
  into the disaggregation.

```python
clean_days = availability.groupby("date")["in_stock"].all()
clean_frame = frame[frame["date"].isin(clean_days[clean_days].index)]
share = (
    clean_frame.groupby("sku_id")["units_sold"].sum()
    / clean_frame["units_sold"].sum()
)
item_forecast = cluster_forecast * share.reindex(cluster_ids).fillna(share.mean())
```

- Recompute the share periodically — it drifts with assortment, pricing, and
  preference changes — rather than freezing it once.
- If item-level models are still needed downstream (for finer features), flag
  rows where any cluster sibling was out of stock (`sibling_oos`) instead of
  deleting them. Deleting creates calendar gaps that models can read as a
  different kind of missingness; flagging lets the model discount the period
  without losing calendar continuity.

## When not to bother

- If no reliable substitute exists for a SKU, don't force a top-down
  decomposition on it — that only adds share-estimation error without a real
  cannibalization problem to solve.
- If stockouts for the cluster are rare or short, an item-level model with a
  stockout-availability feature (see `retail-demand-data`) may already be
  sufficient. Top-down is worth the added complexity mainly when stockouts are
  frequent or long relative to the forecast horizon.

## Required checks before reporting

- substitute cluster membership justified by use-case, price band, and pack
  size, not category alone;
- combined-cluster volume checked to stay roughly stable across stockout
  events, confirming a shift rather than a loss;
- disaggregation share computed only from simultaneous-in-stock days, with the
  estimation window stated;
- item-level training rows flagged, not silently dropped, for sibling-OOS
  periods;
- cluster-level and disaggregated item-level error reported separately;
- share re-estimated on a stated cadence rather than frozen indefinitely.
