---
name: seasonality-transition-effects
description: "Explicitly model the high-error inflection points at season entry and season exit for seasonal verticals (rainwear, AC/coolers, heaters, festive categories), using last year's actual transition dates and a days-into/out-of-season counter. Use whenever forecasting a seasonal vertical, or when a seasonal model's residuals concentrate right at the start or end of the season rather than spreading evenly across it — that concentration is the signature this skill addresses."
---
# Season entry/exit inflection points

Seasonal verticals don't ramp or decay smoothly. The days right around when a
season starts or ends behave the least like the rest of the season and the
least like the immediately preceding off-season, and they consistently carry
the largest share of total seasonal error. A model that captures in-season and
off-season levels well can still fail specifically at these boundaries unless
the transition is modeled as its own regime.

## Diagnose the transition window from history

- Using last year's (and prior years' if available) daily series, find the
  actual date range over which sales rose from the off-season floor to the
  in-season plateau (entry) and fell back down (exit). Don't assume a fixed
  calendar date — entry and exit timing shifts year to year, often driven by
  weather or festival timing.
- Characterize the shape: a sharp step, a multi-day ramp, or an overshoot
  (spike then settle back). This differs by vertical and determines which
  feature form to use.

```python
last_year = frame[frame["date"].between(season_window_start, season_window_end)]
daily = last_year.groupby("date")["units_sold"].sum()
floor = daily.iloc[:7].mean()
plateau = daily.iloc[-7:].mean()
ramp_days = daily[(daily - floor) / max(plateau - floor, 1e-9) < 0.5].index.max()
# ramp_days marks roughly where the transition is still under way, not yet at plateau
```

- Quantify how much error concentrates there: score a plain seasonal-naive or
  trailing-mean model on exactly last year's transition days versus mid-season
  and off-season days. This tells you whether the transition is worth a
  dedicated feature before you build one.

## Model the transition as its own regime

- Add a `days_since_season_start` / `days_since_season_end` feature, clipped
  to a window around each transition (for example `-7..+14`), rather than a
  plain in-season/off-season binary flag. The binary flag treats day 1 of the
  season the same as day 60, which is exactly the distinction that matters
  here.
- Use last year's transition-window demand, indexed by days-from-transition
  rather than by calendar date, as a feature or prior. Aligning by
  days-from-transition lets last year's shape transfer even when this year's
  transition date shifts.
- If this year's transition date is itself uncertain (weather- or
  festival-driven, not yet known), treat transition-date detection as its own
  small task — a leading indicator or a short rolling-window detector — rather
  than forecasting demand purely off a fixed calendar prior.
- For tree/boosting models, add the transition block (days-to-transition,
  last-year-aligned demand, transition-phase flag) together — see
  `feature-selection-tree-models` — and check the ablation specifically on
  last year's transition-window days, not on aggregate error. Aggregate error
  can look fine while masking a large, concentrated miss.

## Evaluation traps

- Aggregate seasonal WMAPE can look acceptable while the 1-2 week transition
  window carries most of the error. Report error split by pre-transition,
  transition-window, plateau, and post-transition separately.
- Waiting for this year's data to confirm the transition has started before
  switching regimes introduces lag: by the time the switch is confident, the
  highest-error days have already passed. A smooth days-from-transition
  feature blended with a leading indicator is usually safer than a hard
  regime switch.

## Required checks before reporting

- transition window (entry and exit) identified from actual prior-year data,
  not assumed from a fixed calendar date;
- days-from-transition feature clipped to a stated window and known/estimable
  at each forecast origin, never derived from this year's future actuals;
- last-year transition-aligned demand used only as a feature/prior, never as
  this year's holdout;
- error reported separately for transition-window days versus
  plateau/off-season days;
- transition-date uncertainty, if present, handled by a leading indicator or
  explicit fallback rather than assumed fixed.
