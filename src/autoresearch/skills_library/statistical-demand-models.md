---
name: statistical-demand-models
description: Implement and debug seasonal naive, ETS/Holt-Winters, ARIMA, and SARIMAX demand forecasts with statsmodels or StatsForecast, including per-series routing, known-future regressors, convergence fallbacks, and runtime controls.
---
# Statistical demand forecasting

Use statistical models when individual series have enough history and most of
their predictable structure is level, trend, seasonality, or autocorrelation.
They are valuable baselines even when a global model is expected to win.

## Establish the baseline ladder

Implement these in order and stop when added complexity does not improve a
horizon-matched temporal backtest:

1. recent mean and seasonal naive (`y[t - seasonal_period]`);
2. damped trend / exponential smoothing;
3. Holt-Winters when stable seasonality is visible;
4. ARIMA for short-memory autocorrelation;
5. SARIMA for genuine seasonal autocorrelation;
6. SARIMAX only when future exogenous values are known.

Do not grid-search large order spaces inside an agent session. For a live
demand task, two or three justified specifications plus a fallback are better
than auto-ARIMA over every SKU.

## Prepare each series correctly

- Sort by timestamp, deduplicate item/date, and reindex to the task frequency.
- Decide what a missing row means. Fill with zero only when it truly means zero
  sales; use missing/imputation when it means no observation or stockout.
- Keep all series on the same request calendar and forecast exactly the
  requested dates.
- Require at least two seasonal cycles before fitting a seasonal component.
  Daily weekly seasonality needs at least 14 observations; annual seasonality
  needs years, not months.
- Fit transformations on history only. If using `log1p`, invert with `expm1`,
  clip at zero, and inspect bias.

## Seasonal naive is a required control

```python
def seasonal_naive(y, horizon, period=7):
    values = y.to_numpy(dtype=float)
    if len(values) < period:
        return np.repeat(np.nanmean(values), horizon)
    return np.resize(values[-period:], horizon)
```

This is hard to beat on stable weekly retail demand and is the correct fallback
for fit failures. A fallback must return the whole horizon, remain finite, and
never read validation actuals.

## ETS and Holt-Winters with statsmodels

Use `statsmodels.tsa.holtwinters.ExponentialSmoothing` for level/trend/seasonal
structure. Additive seasonality is safer when seasonal amplitude is roughly
constant; multiplicative components require strictly positive data.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit = ExponentialSmoothing(
    y,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated",
).fit(optimized=True)
forecast = fit.forecast(horizon)
```

Compare with `trend=None` and with seasonal naive. Do not add both weekly and
annual seasonality through this API; for multiple seasonalities prefer Fourier
regressors with ARIMA errors or a model designed for multiple periods.

## ARIMA and SARIMAX with statsmodels

`ARIMA` is appropriate for non-seasonal ARIMA with optional regressors:

```python
from statsmodels.tsa.arima.model import ARIMA

fit = ARIMA(
    y_train,
    exog=X_train,              # omit when no known-future regressors
    order=(1, 0, 1),
    trend="c",
).fit(method_kwargs={"maxiter": 50})
forecast = fit.forecast(steps=horizon, exog=X_future)
```

Use `SARIMAX` for seasonal orders and stronger state-space controls:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

fit = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 0, 7),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False, maxiter=50)
forecast = fit.forecast(steps=horizon, exog=X_future)
```

Rules that prevent common broken implementations:

- `X_future` is mandatory if the model was fit with `exog`; it must have
  `horizon` rows, identical columns, order, and transformations.
- Calendar indicators and genuinely planned prices/promotions are valid exog.
  Future observed sales, rolling demand over the holdout, and unplanned future
  prices are leakage.
- Weekly seasonality can be represented either by weekday dummies or a
  seasonal AR term. Test both; combining both can be redundant.
- Differencing and a constant/trend interact. Keep order choices small and
  inspect convergence warnings rather than suppressing every exception.
- Clip only after forecasting. A negative statistical forecast is invalid for
  demand but is also diagnostic evidence that the specification is poor.

## Scale to many items

Per-item statsmodels fits are sequential and can dominate runtime. Route models
instead of fitting the most expensive specification everywhere:

- constant/near-constant: constant forecast;
- intermittent: Croston/SBA/TSB skill;
- short history: seasonal naive or recent mean;
- smooth, seasonal: ETS;
- enough dense history with residual autocorrelation: ARIMA/SARIMAX.

Wrap every fit:

```python
try:
    pred = fit_model(y, request_dates)
    if len(pred) != len(request_dates) or not np.isfinite(pred).all():
        raise ValueError("invalid forecast")
except Exception:
    pred = seasonal_naive(y, len(request_dates), period=7)
```

If `statsforecast` is installed, its compiled implementations (`AutoARIMA`,
`AutoETS`, `SeasonalNaive`, Croston variants) are preferable for hundreds or
thousands of series. Verify its long-data schema (`unique_id`, `ds`, `y`) and
installed model names before editing; do not add it solely for a tiny panel.

## Diagnose and tune

- Plot/measure residual autocorrelation after seasonal naive. ARIMA is justified
  only if residual lag structure remains.
- Compare errors by horizon. ARIMA often degrades with horizon while seasonal
  naive remains stable.
- Check AIC/BIC only within the same training sample; promotion is determined
  by out-of-sample task metrics, not information criteria.
- Reject specifications with convergence failure, explosive forecasts, huge
  confidence intervals, or runtime beyond the guardrail.
- Keep model orders and seasonal periods in code as a small explicit candidate
  set, not an opaque expensive search.

## Required checks before reporting

- every requested series/date is covered exactly once;
- future exogenous design matches training design;
- no validation or holdout actual enters fit, order selection, or calibration;
- fit failures and fallback counts are recorded;
- forecasts are finite and non-negative after final clipping;
- metric, signed bias, and runtime beat or contextualize seasonal naive.
