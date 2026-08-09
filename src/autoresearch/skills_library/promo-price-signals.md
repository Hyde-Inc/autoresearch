---
name: promo-price-signals
description: Model demand changes caused by known prices, discounts, and promotions.
---
Treat future price or promotion information as usable only when it is genuinely known at forecast time.

- Add current price, relative price versus recent median, percentage change, discount depth, and promotion duration.
- Estimate uplift globally with SKU or category interactions when individual promotions are sparse.
- Separate baseline demand from incremental promotion lift when promotions create sharp temporary spikes.
- Watch for stockout-censored sales: low observed sales during a promotion may not mean low demand.
- Use regularization or monotonic constraints when price elasticity estimates are unstable.
- Keep a no-promotion fallback for missing future covariates and test whether the feature improves non-promotion dates too.
