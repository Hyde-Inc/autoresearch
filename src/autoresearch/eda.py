"""Exploratory data analysis the Research Director runs before choosing models.

Every function returns a compact, JSON-serializable dict so results can flow
straight back into the director's tool-call loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported data file (need csv or parquet): {path}")


def _round(value: float) -> float:
    return float(np.round(float(value), 4))


def profile(frame: pd.DataFrame) -> dict:
    columns = []
    for name in frame.columns:
        series = frame[name]
        entry: dict = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            entry.update(
                min=_round(series.min()),
                max=_round(series.max()),
                mean=_round(series.mean()),
                zero_share=_round((series.fillna(0) == 0).mean()),
            )
        else:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.9:
                entry["looks_like_dates"] = True
                entry["date_min"] = str(parsed.min().date())
                entry["date_max"] = str(parsed.max().date())
                entry["distinct_dates"] = int(parsed.nunique())
            else:
                entry["examples"] = [str(v) for v in series.dropna().unique()[:5]]
        columns.append(entry)
    return {"rows": len(frame), "columns": columns}


def seasonality(frame: pd.DataFrame, date_column: str, target_column: str) -> dict:
    data = frame[[date_column, target_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column])
    daily = data.groupby(date_column)[target_column].sum().sort_index()
    weekday = daily.groupby(daily.index.dayofweek).mean()
    weekday_profile = {
        day: _round(value / weekday.mean())
        for day, value in zip(
            ["mon", "tue", "wed", "thu", "fri", "sat", "sun"], weekday, strict=False
        )
    }
    autocorr = {
        f"lag_{lag}": _round(daily.autocorr(lag)) if len(daily) > lag + 1 else None
        for lag in (1, 7, 14, 28)
    }
    trend = _round(
        np.polyfit(np.arange(len(daily)), daily.to_numpy(dtype=float), 1)[0]
        / max(daily.mean(), 1e-9)
    )
    return {
        "days": len(daily),
        "weekday_profile_vs_mean": weekday_profile,
        "daily_total_autocorrelation": autocorr,
        "linear_trend_per_day_vs_mean": trend,
    }


def intermittency(
    frame: pd.DataFrame, id_column: str, date_column: str, target_column: str
) -> dict:
    data = frame[[id_column, date_column, target_column]].copy()
    zero_share = data.groupby(id_column)[target_column].apply(
        lambda s: float((s.fillna(0) == 0).mean())
    )
    quadrants = {"smooth": 0, "intermittent": 0, "erratic": 0, "lumpy": 0}
    for _, group in data.groupby(id_column):
        series = group[target_column].fillna(0).to_numpy(dtype=float)
        nonzero = series[series > 0]
        if len(nonzero) == 0:
            quadrants["lumpy"] += 1
            continue
        adi = len(series) / len(nonzero)
        cv2 = float(np.var(nonzero) / max(np.mean(nonzero) ** 2, 1e-12))
        if adi < 1.32 and cv2 < 0.49:
            quadrants["smooth"] += 1
        elif adi >= 1.32 and cv2 < 0.49:
            quadrants["intermittent"] += 1
        elif adi < 1.32:
            quadrants["erratic"] += 1
        else:
            quadrants["lumpy"] += 1
    return {
        "items": int(data[id_column].nunique()),
        "overall_zero_share": _round((data[target_column].fillna(0) == 0).mean()),
        "item_zero_share_quantiles": {
            "p50": _round(zero_share.quantile(0.5)),
            "p90": _round(zero_share.quantile(0.9)),
            "max": _round(zero_share.max()),
        },
        "demand_pattern_counts": quadrants,
    }


def drivers(frame: pd.DataFrame, target_column: str, exclude: list[str]) -> dict:
    target = pd.to_numeric(frame[target_column], errors="coerce")
    correlations: dict[str, float] = {}
    lifts: dict[str, float] = {}
    for name in frame.columns:
        if name == target_column or name in exclude:
            continue
        series = frame[name]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        values = pd.to_numeric(series, errors="coerce")
        unique = set(values.dropna().unique())
        if unique <= {0, 1} and len(unique) > 1:
            on = target[values == 1].mean()
            off = target[values == 0].mean()
            lifts[str(name)] = _round(on / max(off, 1e-9))
        else:
            corr = target.corr(values)
            if pd.notna(corr):
                correlations[str(name)] = _round(corr)
    return {
        "correlation_with_target": correlations,
        "binary_flag_lift_on_target": lifts,
    }
