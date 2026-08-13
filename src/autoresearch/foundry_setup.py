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
    "sales_raw": "sales_raw",
    "sales_train": "sales_train",
    "forecast_request": "forecast_request",
    "forecasts": "forecasts",
    "validation_actuals": "validation_actuals",
    "holdout_actuals": "holdout_actuals",
    "evaluation_metrics": "evaluation_metrics",
}

CONFIG_DATASET_KEYS = (
    "sales_raw",
    "sales_train",
    "forecast_request",
    "forecasts",
    "validation_actuals",
    "holdout_actuals",
    "evaluation_metrics",
)


@dataclass
class SetupResult:
    project_folder_rid: str
    folder_rid: str
    sales_raw: str
    sales_train: str
    forecast_request: str
    forecasts: str
    validation_actuals: str
    holdout_actuals: str
    evaluation_metrics: str


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


def make_raw(train: pd.DataFrame, *, seed: int = 7) -> pd.DataFrame:
    """Dirty up the clean training history so preprocessing has real work to do.

    Injects the classic retail data-quality problems: null demand values,
    missing calendar dates, duplicated rows, and impossible negatives. The
    ``data_preprocessing`` transform on Foundry reverses exactly these."""
    rng = np.random.default_rng(seed)
    raw = train.copy()

    # ~3% of demand values become null.
    nulls = rng.random(len(raw)) < 0.03
    raw.loc[nulls, TARGET] = np.nan

    # ~1% become negative (bad returns handling upstream).
    negatives = rng.random(len(raw)) < 0.01
    raw.loc[negatives, TARGET] = -raw.loc[negatives, TARGET].abs() - 1

    # ~2% of rows vanish entirely (missing dates in the feed).
    keep = rng.random(len(raw)) >= 0.02
    raw = raw[keep]

    # ~1% of the remaining rows are duplicated (double ingestion).
    dupes = raw[rng.random(len(raw)) < 0.01]
    raw = pd.concat([raw, dupes], ignore_index=True)

    return raw.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


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
    """Ensure the pipeline datasets exist and upload the raw inputs.

    Only the *inputs* are uploaded: the dirty ``sales_raw`` feed, the request,
    and the sealed actuals. ``sales_train``, ``forecasts`` and
    ``evaluation_metrics`` are created empty - Foundry builds fill them
    (preprocessing -> model -> evaluation)."""
    project = _resolve_project_folder(repo_rid, project_folder_rid)
    folder = foundry.ensure_folder(FOLDER, project)

    rids: dict[str, str] = {
        key: foundry.ensure_dataset(name, folder) for key, name in DATASET_NAMES.items()
    }

    for key in ("sales_raw", "forecast_request", "validation_actuals", "holdout_actuals"):
        foundry.upload_table(rids[key], frames[key], branch=branch)

    return SetupResult(
        project_folder_rid=project,
        folder_rid=folder,
        **{key: rids[key] for key in CONFIG_DATASET_KEYS},
    )


def result_as_config(result: SetupResult) -> dict[str, str]:
    """The subset of RIDs that belong in the task.yaml ``foundry.datasets`` block."""
    data = asdict(result)
    return {key: data[key] for key in CONFIG_DATASET_KEYS}


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

# Fixed wiring - do not edit. autoresearch foundry-setup wired these datasets.
SALES_TRAIN = "{sales_train}"
FORECAST_REQUEST = "{forecast_request}"
FORECASTS = "{forecasts}"

ID = "{id_col}"
DATE = "{date_col}"
TARGET = "{target_col}"


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

_PREPROCESS_TEMPLATE = '''"""Data Pre Processing — cleans the raw sales feed into the training table.

Runs on Foundry as the first stage of the autoresearch pipeline:
``sales_raw`` (nulls, missing dates, duplicates, negatives) -> ``sales_train``.
This transform is FIXED infrastructure: autoresearch agents never modify it.
"""

from __future__ import annotations

import pandas as pd
from transforms.api import Input, Output, lightweight, transform

SALES_RAW = "{sales_raw}"
SALES_TRAIN = "{sales_train}"

ID = "{id_col}"
DATE = "{date_col}"
TARGET = "{target_col}"


@lightweight
@transform(
    sales_raw=Input(SALES_RAW),
    sales_train=Output(SALES_TRAIN),
)
def compute(sales_raw, sales_train):
    raw = sales_raw.pandas()
    sales_train.write_pandas(clean(raw))


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, repair impossible values, fill nulls, and restore missing dates."""
    frame = raw.copy()
    frame[DATE] = pd.to_datetime(frame[DATE])

    # 1. Drop duplicated (sku, date) rows from double ingestion.
    frame = frame.sort_values([ID, DATE]).drop_duplicates([ID, DATE], keep="first")

    # 2. Negative demand is impossible - clip to zero.
    frame[TARGET] = frame[TARGET].clip(lower=0)

    # 3. Fill null demand with the SKU's same-weekday median (fallback: SKU median, 0).
    frame["_dow"] = frame[DATE].dt.dayofweek
    dow_median = frame.groupby([ID, "_dow"])[TARGET].transform("median")
    sku_median = frame.groupby(ID)[TARGET].transform("median")
    frame[TARGET] = frame[TARGET].fillna(dow_median).fillna(sku_median).fillna(0.0)

    # 4. Re-insert missing calendar dates per SKU (zero demand, carried-forward price).
    pieces = []
    for _, group in frame.groupby(ID, sort=False):
        full = pd.date_range(group[DATE].min(), group[DATE].max(), freq="D")
        piece = group.set_index(DATE).reindex(full)
        piece.index.name = DATE
        piece[ID] = piece[ID].ffill().bfill()
        piece[TARGET] = piece[TARGET].fillna(0.0)
        if "promo" in piece.columns:
            piece["promo"] = piece["promo"].fillna(0).astype(int)
        if "price" in piece.columns:
            piece["price"] = piece["price"].ffill().bfill()
        if "category" in piece.columns:
            piece["category"] = piece["category"].ffill().bfill()
        pieces.append(piece.reset_index())

    result = pd.concat(pieces, ignore_index=True).drop(columns=["_dow"])
    return result.sort_values([ID, DATE]).reset_index(drop=True)
'''

_EVALUATE_TEMPLATE = '''"""Model Evaluation — scores the forecasts against validation actuals on Foundry.

Third stage of the autoresearch pipeline: joins ``forecasts`` with
``validation_actuals`` and writes per-SKU and overall metrics to
``evaluation_metrics`` so results are visible inside Foundry itself.

This transform is FIXED infrastructure: autoresearch agents never modify it.
The model transform never sees these actuals - only this evaluation stage and
the autoresearch CLI do. The sealed holdout split is scored by the CLI alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from transforms.api import Input, Output, lightweight, transform

FORECASTS = "{forecasts}"
VALIDATION_ACTUALS = "{validation_actuals}"
EVALUATION_METRICS = "{evaluation_metrics}"

ID = "{id_col}"
DATE = "{date_col}"
TARGET = "{target_col}"


@lightweight
@transform(
    forecasts=Input(FORECASTS),
    validation_actuals=Input(VALIDATION_ACTUALS),
    evaluation_metrics=Output(EVALUATION_METRICS),
)
def compute(forecasts, validation_actuals, evaluation_metrics):
    scored = score(forecasts.pandas(), validation_actuals.pandas())
    evaluation_metrics.write_pandas(scored)


def _metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    error = forecast - actual
    abs_error = np.abs(error)
    denominator = float(np.abs(actual).sum())
    nonzero = np.abs(actual) > 1e-8
    return {{
        "wmape": float(abs_error.sum() / denominator) if denominator else float("nan"),
        "mape": float(np.mean(abs_error[nonzero] / np.abs(actual[nonzero]))) if nonzero.any() else 0.0,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias_pct": float(error.sum() / denominator * 100) if denominator else float("nan"),
        "n_rows": int(len(actual)),
    }}


def score(forecasts: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    forecasts = forecasts.copy()
    actuals = actuals.copy()
    forecasts[DATE] = pd.to_datetime(forecasts[DATE])
    actuals[DATE] = pd.to_datetime(actuals[DATE])
    merged = actuals.merge(forecasts[[ID, DATE, "forecast"]], on=[ID, DATE], how="inner")

    rows = [{{"scope": "overall", ID: "ALL", **_metrics(
        merged[TARGET].to_numpy(dtype=float), merged["forecast"].to_numpy(dtype=float))}}]
    for sku, group in merged.groupby(ID):
        rows.append({{"scope": "sku", ID: sku, **_metrics(
            group[TARGET].to_numpy(dtype=float), group["forecast"].to_numpy(dtype=float))}})
    return pd.DataFrame(rows)
'''

_CONTRACT_TEMPLATE = """# Autoresearch × Foundry

This Foundry transforms repository is driven by the **autoresearch CLI**: the research
loop runs on your machine, but every training run is a **Foundry build**. Code here,
build there, iterate.

## The pipeline (three stages, three folders)

```
sales_raw ──▶ [data_preprocessing] ──▶ sales_train ──▶ [model_running] ──▶ forecasts
                                                                              │
                              evaluation_metrics ◀── [model_evaluation] ◀────┘
```

1. **`data_preprocessing/preprocess.py`** — fixed infrastructure. Cleans the raw feed:
   deduplicates rows, clips negative demand, fills null demand (same-weekday median),
   restores missing calendar dates.
2. **`model_running/forecast.py`** — the ONLY file autoresearch agents edit. Reads
   `sales_train` + `forecast_request`, writes `forecasts`.
3. **`model_evaluation/evaluate.py`** — fixed infrastructure. Joins `forecasts` with
   `validation_actuals` and writes per-SKU + overall metrics to `evaluation_metrics`,
   so results are visible inside Foundry.

## The research loop

1. BUILD — an agent rewrites `build_forecasts()` in `{transform_rel}`.
2. TRAIN — the CLI pushes the experiment to its own Foundry branch; Foundry publishes
   the transform and builds `forecasts` on that branch (inputs fall back to `{branch}`),
   so parallel agents never collide.
3. EVALUATE — the CLI reads the branch's `forecasts` plus the sealed actuals via
   readTable, scores them, and feeds the result back to the agent.
4. PROMOTE — the best passing experiment is merged and pushed to `{branch}`.

Validation drives the improve loop; the hidden holdout is scored only at the final gate.

## Datasets ({data_home})

| Dataset             | Role                            | Visible to the model transform? |
| ------------------- | ------------------------------- | ------------------------------- |
| `sales_raw`         | dirty upstream feed             | no — preprocessing only         |
| `sales_train`       | cleaned training history        | yes                             |
| `forecast_request`  | (id, date) rows to score        | yes                             |
| `forecasts`         | model output (build result)     | written by the build            |
| `validation_actuals`| sealed validation answers       | **no** — evaluation + CLI only  |
| `holdout_actuals`   | sealed holdout answers          | **no** — CLI only               |
| `evaluation_metrics`| per-item + overall scores       | written by evaluation           |

`sales_train` schema: `{id_col}` (str), `{date_col}` (date), `{target_col}` (float),
plus any feature columns present in the dataset.
`forecasts` schema (exact): `{id_col}` (str), `{date_col}` (date), `forecast` (float,
finite, ≥ 0), one row per requested (`{id_col}`, `{date_col}`).

## Rules for the agent

- Edit **only** `build_forecasts()` in `{transform_rel}`.
- Do **not** touch `data_preprocessing/` or `model_evaluation/` — fixed infrastructure.
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


# Each stage lists the dataset refs its template needs; a stage is only
# scaffolded when all of them are wired (e.g. no sales_raw -> no preprocessing).
STAGES = {
    "data_preprocessing": ("preprocess.py", _PREPROCESS_TEMPLATE, ("sales_raw", "sales_train")),
    "model_running": (
        "forecast.py",
        _TRANSFORM_TEMPLATE,
        ("sales_train", "forecast_request", "forecasts"),
    ),
    "model_evaluation": (
        "evaluate.py",
        _EVALUATE_TEMPLATE,
        ("forecasts", "validation_actuals", "evaluation_metrics"),
    ),
}


def default_refs(datasets_path: str) -> dict[str, str]:
    """Dataset path refs for the provisioned layout under ``datasets_path``."""
    return {key: f"{datasets_path}/{name}" for key, name in DATASET_NAMES.items()}


def scaffold(
    repo: Path,
    refs: dict[str, str],
    branch: str,
    *,
    id_col: str = "sku_id",
    date_col: str = "date",
    target_col: str = "units_sold",
    data_home: str = "",
) -> tuple[str, list[str]]:
    """Install the autoresearch pipeline into ``repo``.

    ``refs`` maps dataset keys to the Input/Output references the transforms
    are wired with - dataset paths for the provisioned demo layout, or raw
    RIDs when plugging into existing datasets. Stages whose datasets are not
    wired are skipped. Each file is written only if missing so an improved
    model is never clobbered; a legacy single-file ``datasets/forecast.py``
    is migrated into ``model_running/`` so its logic survives.
    Also writes the AUTORESEARCH.md contract and the curated dependencies.
    Returns ``(model_transform_rel_path, actions)``."""
    package = find_package_dir(repo)
    datasets_dir = package / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    actions: list[str] = []
    legacy = datasets_dir / "forecast.py"
    columns = {"id_col": id_col, "date_col": date_col, "target_col": target_col}
    for folder, (filename, template, needed) in STAGES.items():
        if not all(refs.get(key) for key in needed):
            continue
        stage_dir = datasets_dir / folder
        stage_dir.mkdir(exist_ok=True)
        init = stage_dir / "__init__.py"
        if not init.exists():
            init.write_text("")
        target = stage_dir / filename
        if target.exists():
            continue
        if folder == "model_running" and legacy.exists():
            target.write_text(legacy.read_text())
            legacy.unlink()
            actions.append("migrated existing model into model_running/forecast.py")
        else:
            fields = {key: refs.get(key, "") for key in DATASET_NAMES}
            target.write_text(template.format(**fields, **columns))
            actions.append(f"wrote {folder}/{filename}")
    if legacy.exists():
        # model_running/forecast.py already existed: the legacy copy would
        # double-register the forecasts output and break the publish.
        legacy.unlink()
        actions.append("removed legacy datasets/forecast.py duplicate")
    transform_rel = str((datasets_dir / "model_running" / "forecast.py").relative_to(repo))
    contract = repo / "AUTORESEARCH.md"
    contract.write_text(
        _CONTRACT_TEMPLATE.format(
            data_home=data_home or "RIDs wired into the transforms",
            transform_rel=transform_rel,
            branch=branch,
            **columns,
        )
    )
    actions.append("wrote AUTORESEARCH.md")
    if _patch_conda_recipe(repo):
        actions.append("added forecasting packages to conda_recipe/meta.yaml")
    return transform_rel, actions
