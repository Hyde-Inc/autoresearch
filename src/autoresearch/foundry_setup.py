"""Phase 0: put the demo data on Foundry so training can run there.

Generates a synthetic daily demand history, splits it chronologically into
train / validation / holdout, then provisions five Foundry datasets under the
project's ``autoresearch`` folder:

    sales_train         training history (model input)
    forecast_request    (sku_id, date) rows to predict — validation + holdout
    forecasts           empty output the build fills (pre-created so we have its RID)
    validation_actuals  sealed validation answers (CLI reads these, model never does)
    holdout_actuals     sealed holdout answers

The transform in the repo binds to these by path, so the paths here must match
``datasets/forecast.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import foundry

ID = "sku_id"
DATE = "date"
TARGET = "units_sold"

FOLDER = "autoresearch"
DATASET_NAMES = {
    "sales_train": "sales_train",
    "forecast_request": "forecast_request",
    "forecasts": "forecasts",
    "validation_actuals": "validation_actuals",
    "holdout_actuals": "holdout_actuals",
}


@dataclass
class SetupResult:
    project_folder_rid: str
    folder_rid: str
    sales_train: str
    forecast_request: str
    forecasts: str
    validation_actuals: str
    holdout_actuals: str


def generate_history(
    *, seed: int = 42, n_skus: int = 10, n_intermittent: int = 2, n_days: int = 210
) -> pd.DataFrame:
    """Daily sales for ``n_skus`` SKUs with weekly/annual seasonality, promos,
    price moves, and a few intermittent slow movers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    day = np.arange(n_days)
    frames: list[pd.DataFrame] = []
    for index in range(n_skus):
        intermittent = index < n_intermittent
        promo = rng.random(n_days) < rng.uniform(0.02, 0.08)
        price = rng.uniform(5, 100) * (1 - promo * rng.uniform(0.05, 0.25))
        if intermittent:
            buy_probability = rng.uniform(0.05, 0.25)
            demand = np.where(
                rng.random(n_days) < buy_probability * (1 + promo * 0.8),
                rng.poisson(rng.uniform(1, 4), n_days),
                0,
            ).astype(float)
        else:
            base = rng.uniform(8, 80)
            weekly_amp = rng.uniform(0.08, 0.35)
            annual_amp = rng.uniform(0.03, 0.20)
            trend = rng.uniform(-0.00025, 0.0008)
            phase = rng.uniform(0, 2 * np.pi)
            signal = (
                base
                * (1 + trend * day)
                * (1 + weekly_amp * np.sin(2 * np.pi * day / 7 + phase))
                * (1 + annual_amp * np.sin(2 * np.pi * day / 365 + phase / 2))
                * (1 + promo * rng.uniform(0.15, 0.65))
            )
            noise = rng.normal(0, np.sqrt(np.maximum(signal, 1)) * 0.8)
            demand = np.maximum(0, np.round(signal + noise))
        frames.append(
            pd.DataFrame(
                {
                    ID: f"SKU_{index:03d}",
                    DATE: dates,
                    TARGET: demand.astype(float),
                    "promo": promo.astype(int),
                    "price": price.round(2),
                    "category": f"CAT_{index % 6:02d}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def chronological_split(
    frame: pd.DataFrame, *, validation_days: int = 28, holdout_days: int = 28
) -> dict[str, pd.DataFrame]:
    """Split by date: last ``holdout_days`` → holdout, the ``validation_days``
    before that → validation, everything earlier → train."""
    frame = frame.copy()
    frame[DATE] = pd.to_datetime(frame[DATE])
    unique_dates = np.sort(frame[DATE].unique())
    if len(unique_dates) <= validation_days + holdout_days:
        raise ValueError("history too short for the requested validation/holdout windows")
    holdout_start = unique_dates[-holdout_days]
    validation_start = unique_dates[-(holdout_days + validation_days)]

    train = frame[frame[DATE] < validation_start]
    validation = frame[(frame[DATE] >= validation_start) & (frame[DATE] < holdout_start)]
    holdout = frame[frame[DATE] >= holdout_start]

    keys = [ID, DATE]
    validation_actuals = validation[[*keys, TARGET]].reset_index(drop=True)
    holdout_actuals = holdout[[*keys, TARGET]].reset_index(drop=True)
    request = pd.concat([validation[keys], holdout[keys]], ignore_index=True)
    return {
        "sales_train": train.reset_index(drop=True),
        "forecast_request": request.reset_index(drop=True),
        "validation_actuals": validation_actuals,
        "holdout_actuals": holdout_actuals,
    }


def _resolve_project_folder(repo_rid: str, project_folder_rid: str) -> str:
    if project_folder_rid:
        return project_folder_rid
    if not repo_rid:
        raise foundry.FoundryError("provide either project_folder_rid or repo_rid for setup")
    return foundry.resolve_parent_folder(repo_rid)


def provision(
    frames: dict[str, pd.DataFrame],
    *,
    repo_rid: str = "",
    project_folder_rid: str = "",
    branch: str = "master",
) -> SetupResult:
    """Ensure the five datasets exist and upload the four input tables."""
    project = _resolve_project_folder(repo_rid, project_folder_rid)
    folder = foundry.ensure_folder(FOLDER, project)

    rids: dict[str, str] = {
        key: foundry.ensure_dataset(name, folder) for key, name in DATASET_NAMES.items()
    }

    for key in ("sales_train", "forecast_request", "validation_actuals", "holdout_actuals"):
        foundry.upload_table(rids[key], frames[key], branch=branch)

    return SetupResult(
        project_folder_rid=project,
        folder_rid=folder,
        sales_train=rids["sales_train"],
        forecast_request=rids["forecast_request"],
        forecasts=rids["forecasts"],
        validation_actuals=rids["validation_actuals"],
        holdout_actuals=rids["holdout_actuals"],
    )


def result_as_config(result: SetupResult) -> dict[str, str]:
    """The subset of RIDs that belong in the task.yaml ``foundry.datasets`` block."""
    data = asdict(result)
    return {
        key: data[key]
        for key in (
            "sales_train",
            "forecast_request",
            "forecasts",
            "validation_actuals",
            "holdout_actuals",
        )
    }
