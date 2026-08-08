"""Error analysis over saved validation forecasts.

The harness stores each scored attempt's merged validation frame (actuals plus
forecasts). These tools let the Research Director study where a model is wrong
- by item, weekday, and bias direction - before deciding what to try next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .harness import forecasting_metrics


def _round(value: float) -> float:
    return float(np.round(float(value), 4))


def error_summary(
    frame: pd.DataFrame, id_column: str, date_column: str, target_column: str
) -> dict:
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    actual = frame[target_column].to_numpy(dtype=float)
    forecast = frame["forecast"].to_numpy(dtype=float)
    overall = {k: _round(v) for k, v in forecasting_metrics(actual, forecast).items()}

    frame["abs_error"] = np.abs(forecast - actual)
    frame["error"] = forecast - actual
    by_weekday = {}
    for day, group in frame.groupby(frame[date_column].dt.dayofweek):
        label = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][int(day)]
        denominator = max(group[target_column].abs().sum(), 1e-9)
        by_weekday[label] = {
            "wmape": _round(group["abs_error"].sum() / denominator),
            "bias_pct": _round(group["error"].sum() / denominator * 100),
        }
    by_horizon = {}
    for position, (_, group) in enumerate(frame.groupby(frame[date_column]), start=1):
        denominator = max(group[target_column].abs().sum(), 1e-9)
        by_horizon[position] = _round(group["abs_error"].sum() / denominator)
    horizon_series = pd.Series(by_horizon)
    item_bias = frame.groupby(id_column)["error"].sum()
    return {
        "overall": overall,
        "wmape_by_weekday": by_weekday,
        "wmape_first_vs_last_horizon_third": {
            "first_third": _round(horizon_series.head(max(1, len(horizon_series) // 3)).mean()),
            "last_third": _round(horizon_series.tail(max(1, len(horizon_series) // 3)).mean()),
        },
        "items_over_forecast": int((item_bias > 0).sum()),
        "items_under_forecast": int((item_bias < 0).sum()),
    }


def worst_items(
    frame: pd.DataFrame,
    id_column: str,
    date_column: str,
    target_column: str,
    limit: int = 10,
) -> dict:
    frame = frame.copy()
    frame["abs_error"] = np.abs(frame["forecast"] - frame[target_column])
    frame["error"] = frame["forecast"] - frame[target_column]
    total_abs_error = max(frame["abs_error"].sum(), 1e-9)
    grouped = frame.groupby(id_column).agg(
        actual_total=(target_column, "sum"),
        abs_error=("abs_error", "sum"),
        bias=("error", "sum"),
    )
    grouped["error_share"] = grouped["abs_error"] / total_abs_error
    grouped["wmape"] = grouped["abs_error"] / grouped["actual_total"].clip(lower=1e-9)
    top = grouped.sort_values("abs_error", ascending=False).head(limit)
    return {
        "total_items": int(len(grouped)),
        "worst": [
            {
                "item": str(item),
                "actual_total": _round(row.actual_total),
                "wmape": _round(row.wmape),
                "share_of_all_error": _round(row.error_share),
                "direction": "over" if row.bias > 0 else "under",
            }
            for item, row in top.iterrows()
        ],
    }
