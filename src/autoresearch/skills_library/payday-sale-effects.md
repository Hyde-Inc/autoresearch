---
name: payday-sale-effects
description: "Model the spike-then-drop that staple verticals (wheat flour, detergent, rice, dal, edible oil) show around salary payday — a demand shift concentrated in the last 2 days of the month and first 3 days of the next, followed by a give-back. Use whenever forecasting staple/pantry-stocking SKUs, quick-commerce verticals, or when residuals show a recurring month-boundary bias; the payday window pulls volume forward rather than creating it."
---
# Payday demand spikes in staple verticals

Staple, repeat-purchase verticals see a reliable spike around salary payday and
a roughly equal drop in the days right after. Most of that spike is stock-up
buying pulled forward from the following week, not incremental demand. Treat
the payday window as a demand-*shifting* event, the same way you would treat a
promotion, not as a source of new demand.

## Diagnose before modeling

- Start from a default payday window of the last 2 calendar days of the month
  plus the first 3 days of the next (5 days). Confirm against a local salary
  calendar when one is available; government/large-employer payday timing can
  differ by geography.
- Compute an uplift ratio and a trough ratio per item/vertical from training
  history, not assumed uniformly:

```python
frame["day_of_cycle"] = np.where(
    frame["date"].dt.is_month_end | (frame["date"].dt.day <= 2),
    "payday_pre_or_day",
    np.where(frame["date"].dt.day.between(1, 3), "payday_post", "other"),
)
# clean this into a single categorical: last 2 days of month, first 3 of next
baseline = frame.loc[frame["day_of_cycle"] == "other"].groupby("sku_id")["units_sold"].mean()
uplift = frame.loc[frame["day_of_cycle"].str.startswith("payday_pre")].groupby("sku_id")["units_sold"].mean() / baseline
trough = frame.loc[frame["day_of_cycle"] == "payday_post"].groupby("sku_id")["units_sold"].mean() / baseline
```

- Segment verticals: pantry-stocking staples (atta, detergent, rice, dal, oil)
  typically show strong uplift and a matching trough; perishables and impulse
  categories usually don't. Don't apply the payday feature to a vertical
  without first checking its own uplift/trough ratio.

## Model the pattern explicitly

- Add a deterministic calendar feature — signed distance to the nearest month
  boundary (for example `-2..+3`) — rather than a plain binary payday flag.
  The signed distance lets a model learn the shape of the ramp and the
  give-back, not just "on" vs "off".
- For statistical baselines, a 28-day seasonal period will not align with
  variable month lengths, so it cannot absorb this pattern reliably. Represent
  the window as an explicit exogenous regressor in ARIMAX/SARIMAX rather than
  hoping a fixed seasonal period captures it.
- For global panel/boosting models, add the payday feature together with its
  vertical-elasticity interaction as one block (signed day-of-cycle,
  `is_staple_vertical`, and their interaction) — see
  `feature-selection-tree-models` for the block-testing procedure. A lone
  day-of-cycle feature without the vertical interaction will get diluted
  across verticals where the effect doesn't exist.
- Because this is a shift and not new demand, check that the modeled spike and
  the modeled trough roughly net out at the monthly total. A model that
  forecasts the spike but leaves the following week's baseline unchanged is
  double-counting the pulled-forward volume.

## Evaluation traps

- Aggregate WMAPE can look fine while the payday-window days and the
  post-payday trough carry large, offsetting errors. Report error split by
  payday-window, post-payday-trough, and remaining days separately.
- Confirm the uplift is actually payday-driven and not a coincident promotion
  or festival that happens to land near a month boundary; check calendar
  overlap before attributing the spike to payday.

## Required checks before reporting

- payday window definition and source (default 5-day window vs local salary
  calendar) stated explicitly;
- uplift and trough ratios measured per vertical/item, not assumed uniform;
- the feature is deterministic/known-future and never reads realized sales
  inside the window it is predicting;
- error reported separately for payday-window, post-payday-trough, and
  remaining days;
- modeled spike and modeled trough checked to roughly offset at the monthly
  total level;
- no coincident promotion/festival is silently driving the measured uplift.
