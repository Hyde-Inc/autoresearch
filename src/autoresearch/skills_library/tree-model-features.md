---
name: tree-model-features
description: Build leakage-safe lag, rolling, calendar, price, and promotion features for global tree models.
---
For every forecast origin, features must be knowable at prediction time.

- Use shifted lags at seasonal and short horizons, then rolling mean, median, standard deviation, min, and max over shifted demand.
- Add day-of-week, week-of-year, month, holidays when available, SKU identity, price level, price change, and promotion flags.
- Fit one global model across SKUs when data per SKU is limited; encode SKU categorically or with stable integer IDs.
- For multi-step horizons, use direct horizon-specific models or recursive prediction with forecast values fed back. Never let future actuals enter lag features.
- Validate feature availability over the entire requested horizon and retain a seasonal-naive fallback.
