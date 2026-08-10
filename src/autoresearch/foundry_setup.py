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
from pathlib import Path

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


# --------------------------------------------------------------------------
# Repo scaffolding: turn ANY Foundry Python transforms repo into an
# autoresearch-ready one (transform + contract doc + curated dependencies).
# --------------------------------------------------------------------------

REQUIRED_PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "statsmodels", "xgboost", "lightgbm"]


def read_gradle_properties(repo: Path) -> dict[str, str]:
    """Parse ``gradle.properties`` — every transforms repo carries its own
    identity there (repo RID, project path, default branch)."""
    path = repo / "gradle.properties"
    props: dict[str, str] = {}
    if not path.exists():
        return props
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


def find_package_dir(repo: Path) -> Path:
    """The python package under transforms-python/src (usually ``myproject``)."""
    src = repo / "transforms-python" / "src"
    if not src.exists():
        raise foundry.FoundryError(
            f"{repo} does not look like a Foundry Python transforms repo "
            "(transforms-python/src is missing)"
        )
    for child in sorted(src.iterdir()):
        if child.is_dir() and not child.name.endswith(".egg-info") and child.name != "tests":
            return child
    raise foundry.FoundryError(f"no python package found under {src}")


_TRANSFORM_TEMPLATE = '''"""Autoresearch training transform — the model runs on Foundry compute.

The autoresearch loop rewrites ONLY the body of ``build_forecasts`` to improve
the model. Do NOT change the ``@transform`` decorator, the Input/Output dataset
paths, or the output schema: the CLI pushes this repo, triggers a Foundry build
of the ``forecasts`` dataset, then reads it back and scores it against sealed
validation / holdout actuals it alone can see.

Contract
--------
Input  ``sales_train``      : the training history. Columns:
    sku_id (str), date (date), units_sold (float), promo (int 0/1),
    price (float), category (str).
Input  ``forecast_request`` : the rows to forecast (validation + holdout dates),
    columns sku_id (str), date (date). NO target — actuals are sealed.
Output ``forecasts``        : exactly sku_id (str), date (date),
    forecast (float, finite, >= 0). One row per requested (sku_id, date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from transforms.api import Input, Output, lightweight, transform

# Fixed wiring - do not edit. autoresearch foundry-setup created these datasets.
SALES_TRAIN = "{datasets_path}/sales_train"
FORECAST_REQUEST = "{datasets_path}/forecast_request"
FORECASTS = "{datasets_path}/forecasts"

ID = "sku_id"
DATE = "date"
TARGET = "units_sold"


@lightweight
@transform(
    sales_train=Input(SALES_TRAIN),
    forecast_request=Input(FORECAST_REQUEST),
    forecasts=Output(FORECASTS),
)
def compute(sales_train, forecast_request, forecasts):
    train = sales_train.pandas()
    request = forecast_request.pandas()
    result = build_forecasts(train, request)
    forecasts.write_pandas(result)


def build_forecasts(train: pd.DataFrame, request: pd.DataFrame) -> pd.DataFrame:
    """Baseline: per-SKU mean of the same weekday over the last 28 days.

    Deliberately simple - autoresearch replaces this function with stronger
    models. Must return exactly ``request`` rows plus a finite, non-negative
    ``forecast`` column.
    """
    train = train.copy()
    train[DATE] = pd.to_datetime(train[DATE])
    request = request.copy()
    request[DATE] = pd.to_datetime(request[DATE])

    recent = train[train[DATE] >= train[DATE].max() - pd.Timedelta(days=27)]
    recent = recent.assign(_dow=recent[DATE].dt.dayofweek)
    dow_mean = recent.groupby([ID, "_dow"])[TARGET].mean().to_dict()
    sku_mean = recent.groupby(ID)[TARGET].mean().to_dict()
    global_mean = float(train[TARGET].mean()) if len(train) else 0.0

    values = [
        dow_mean.get(
            (getattr(row, ID), getattr(row, DATE).dayofweek),
            sku_mean.get(getattr(row, ID), global_mean),
        )
        for row in request.itertuples(index=False)
    ]
    result = request[[ID, DATE]].copy()
    result["forecast"] = np.maximum(0.0, np.asarray(values, dtype=float))
    return result
'''

_CONTRACT_TEMPLATE = """# Autoresearch × Foundry

This Foundry transforms repository is driven by the **autoresearch CLI**. Agents rewrite
the model code; the actual **training runs on Foundry** as a build of the `forecasts`
dataset. Each experiment is pushed to its own Foundry branch and built there, so parallel
agents produce independent forecasts; the winner is merged and pushed back to `{branch}`.

## The loop

1. BUILD — an agent rewrites `build_forecasts()` in `{transform_rel}`.
2. TRAIN — the CLI pushes the experiment branch, Foundry publishes the transform, and a
   build of `forecasts` runs on that branch (inputs fall back to `{branch}`).
3. EVALUATE — the CLI reads the branch's `forecasts` plus the sealed actuals via
   readTable, scores them, and feeds the result back to the agent.
4. PROMOTE — the best passing experiment is merged and pushed to `{branch}`.

Validation drives the improve loop; the hidden holdout is scored only at the final gate.

## Datasets (under `{datasets_path}/`)

| Dataset             | Role                          | Visible to the transform? |
| ------------------- | ----------------------------- | ------------------------- |
| `sales_train`       | training history (input)      | yes                       |
| `forecast_request`  | (sku_id, date) rows to score  | yes                       |
| `forecasts`         | model output (build result)   | written by the build      |
| `validation_actuals`| sealed validation answers     | **no** — CLI only         |
| `holdout_actuals`   | sealed holdout answers        | **no** — CLI only         |

`sales_train` schema: `sku_id` (str), `date` (date), `units_sold` (float), `promo` (0/1),
`price` (float), `category` (str).
`forecasts` schema (exact): `sku_id` (str), `date` (date), `forecast` (float, finite, ≥ 0),
one row per requested (sku_id, date).

## Rules for the agent

- Edit **only** `build_forecasts()` in `{transform_rel}`.
- Do **not** change the `@transform` decorator, the Input/Output paths, or the output schema.
- Do **not** edit dependencies. The forecasting stack (numpy, pandas, scikit-learn,
  statsmodels, xgboost, lightgbm) is pre-declared in `transforms-python/conda_recipe/meta.yaml`.
- The model may read only `sales_train` and `forecast_request`; the actuals are never inputs.
"""


def _patch_conda_recipe(repo: Path) -> bool:
    """Add the curated forecasting packages to the recipe's ``run:`` block.

    meta.yaml contains jinja placeholders, so this is a text patch, not a yaml
    round-trip. Returns True when the file changed."""
    path = repo / "transforms-python" / "conda_recipe" / "meta.yaml"
    if not path.exists():
        return False
    text = path.read_text()
    missing = [pkg for pkg in REQUIRED_PACKAGES if f"- {pkg}" not in text]
    if not missing:
        return False
    lines = text.splitlines()
    # Insert after the last "    - ..." entry of the run: block.
    insert_at = None
    in_run = False
    for index, line in enumerate(lines):
        if line.strip() == "run:":
            in_run = True
            insert_at = index
            continue
        if in_run:
            if line.strip().startswith("- ") or line.strip().startswith("#") or not line.strip():
                if line.strip().startswith("- "):
                    insert_at = index
            else:
                break
    if insert_at is None:
        return False
    addition = ["    # Forecasting stack for the autoresearch model transform."] + [
        f"    - {pkg}" for pkg in missing
    ]
    lines[insert_at + 1 : insert_at + 1] = addition
    path.write_text("\n".join(lines) + "\n")
    return True


def scaffold(repo: Path, datasets_path: str, branch: str) -> tuple[str, list[str]]:
    """Install the autoresearch contract into ``repo``.

    Writes the model transform (only if missing, so an improved model is never
    clobbered), the AUTORESEARCH.md contract, and the curated dependencies.
    Returns ``(transform_rel_path, actions)``."""
    package = find_package_dir(repo)
    datasets_dir = package / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    transform = datasets_dir / "forecast.py"
    transform_rel = str(transform.relative_to(repo))
    actions: list[str] = []
    if not transform.exists():
        transform.write_text(_TRANSFORM_TEMPLATE.format(datasets_path=datasets_path))
        actions.append(f"wrote baseline transform {transform_rel}")
    contract = repo / "AUTORESEARCH.md"
    contract.write_text(
        _CONTRACT_TEMPLATE.format(
            datasets_path=datasets_path, transform_rel=transform_rel, branch=branch
        )
    )
    actions.append("wrote AUTORESEARCH.md")
    if _patch_conda_recipe(repo):
        actions.append("added forecasting packages to conda_recipe/meta.yaml")
    return transform_rel, actions
