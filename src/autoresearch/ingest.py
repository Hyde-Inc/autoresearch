"""CSV ingestion: turn a standard-schema sales export into a runnable task.

The product's fixed input schema is:

    date, sku_name, sales, selling_price

``preview_csv`` inspects an upload without touching disk so the UI can validate
and show the user what was found. ``ingest_csv`` cleans the data, splits it
chronologically into train / validation / holdout, and writes a complete task
directory (seed workspace, private actuals, and task.yaml) that the existing
orchestrator can run unchanged.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .config import DEFAULT_MODEL

REQUIRED_COLUMNS = ["date", "sku_name", "sales", "selling_price"]
ID_COLUMN = "sku_name"
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
NUMERIC_COLUMNS = ["sales", "selling_price"]


@dataclass
class Preview:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    skus: int = 0
    date_min: str | None = None
    date_max: str | None = None
    distinct_dates: int = 0
    sample: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_validation_days: int = 0
    suggested_holdout_days: int = 0


@dataclass
class IngestResult:
    task: str
    config_path: Path
    task_dir: Path
    train_rows: int
    validation_rows: int
    holdout_rows: int
    skus: int
    train_end: str
    validation_end: str
    holdout_end: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "task"


def _read_csv(data: str | bytes | Path) -> pd.DataFrame:
    if isinstance(data, Path):
        return pd.read_csv(data)
    if isinstance(data, bytes):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(io.StringIO(data))


def _clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalise types, drop unusable rows, and report what was adjusted."""
    warnings: list[str] = []
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]

    parsed = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    bad_dates = int(parsed.isna().sum())
    if bad_dates:
        warnings.append(f"dropped {bad_dates} rows with unparseable dates")
    frame[DATE_COLUMN] = parsed

    for column in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(frame[column], errors="coerce")
        bad = int(coerced.isna().sum())
        if bad:
            warnings.append(f"dropped {bad} rows with non-numeric {column}")
        frame[column] = coerced

    frame[ID_COLUMN] = frame[ID_COLUMN].astype(str).str.strip()

    before = len(frame)
    frame = frame.dropna(subset=REQUIRED_COLUMNS)
    frame = frame[frame[ID_COLUMN] != ""]

    negatives = int((frame[TARGET_COLUMN] < 0).sum())
    if negatives:
        warnings.append(f"clipped {negatives} negative {TARGET_COLUMN} values to 0")
        frame.loc[frame[TARGET_COLUMN] < 0, TARGET_COLUMN] = 0.0

    dupes = frame.duplicated([ID_COLUMN, DATE_COLUMN], keep="last")
    if int(dupes.sum()):
        warnings.append(f"collapsed {int(dupes.sum())} duplicate sku_name/date rows (kept last)")
        frame = frame[~dupes]

    dropped = before - len(frame)
    if dropped and not warnings:
        warnings.append(f"dropped {dropped} unusable rows")

    frame = frame.sort_values([ID_COLUMN, DATE_COLUMN]).reset_index(drop=True)
    return frame, warnings


def _suggest_horizon(distinct_dates: int) -> int:
    """A reasonable per-split horizon that still leaves the bulk for training."""
    if distinct_dates < 6:
        return 1
    return max(1, min(28, distinct_dates // 5))


def preview_csv(data: str | bytes | Path) -> Preview:
    try:
        frame = _read_csv(data)
    except Exception as exc:  # noqa: BLE001
        return Preview(ok=False, errors=[f"could not read CSV: {exc}"])

    columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        return Preview(
            ok=False,
            columns=columns,
            rows=len(frame),
            errors=[f"missing required column(s): {', '.join(missing)}"],
        )

    cleaned, warnings = _clean(frame)
    if cleaned.empty:
        return Preview(
            ok=False,
            columns=columns,
            rows=len(frame),
            errors=["no usable rows remain after cleaning"],
            warnings=warnings,
        )

    distinct_dates = int(cleaned[DATE_COLUMN].nunique())
    errors: list[str] = []
    if distinct_dates < 4:
        errors.append(
            f"only {distinct_dates} distinct dates found; need at least 4 to split "
            "train/validation/holdout"
        )
    horizon = _suggest_horizon(distinct_dates)
    sample = (
        cleaned.head(8)
        .assign(**{DATE_COLUMN: cleaned.head(8)[DATE_COLUMN].dt.strftime("%Y-%m-%d")})
        .to_dict(orient="records")
    )
    return Preview(
        ok=not errors,
        columns=columns,
        rows=len(cleaned),
        skus=int(cleaned[ID_COLUMN].nunique()),
        date_min=cleaned[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        date_max=cleaned[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        distinct_dates=distinct_dates,
        sample=sample,
        errors=errors,
        warnings=warnings,
        suggested_validation_days=horizon,
        suggested_holdout_days=horizon,
    )


_BASELINE_TEMPLATE = '''\
"""Seasonal-naive baseline: repeat the same weekday from the most recent week."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID = "{id_column}"
DATE = "{date_column}"
TARGET = "{target_column}"


def main() -> None:
    train = pd.read_parquet(os.environ["AUTORESEARCH_TRAIN_DATA"])
    request = pd.read_parquet(os.environ["AUTORESEARCH_REQUEST"])
    output = Path(os.environ["AUTORESEARCH_OUTPUT"])
    train[DATE] = pd.to_datetime(train[DATE])
    request[DATE] = pd.to_datetime(request[DATE])

    history = {{
        key: group.set_index(DATE)[TARGET].sort_index()
        for key, group in train.groupby(ID, sort=False)
    }}
    global_mean = float(train[TARGET].mean())
    means = train.groupby(ID)[TARGET].mean().to_dict()

    forecasts = []
    for row in request.itertuples(index=False):
        key = getattr(row, ID)
        series = history.get(key)
        value = means.get(key, global_mean)
        if series is not None and len(series):
            lag_date = getattr(row, DATE)
            steps = 0
            while lag_date not in series.index and steps < 60:
                lag_date -= pd.Timedelta(days=7)
                steps += 1
            value = float(series.loc[lag_date]) if lag_date in series.index else float(series.iloc[-1])
        forecasts.append(max(0.0, float(value)))

    result = request[[ID, DATE]].copy()
    result["forecast"] = np.asarray(forecasts, dtype=float)
    result.to_parquet(output, index=False)


if __name__ == "__main__":
    main()
'''

_TASK_MD_TEMPLATE = """\
# {title}

Improve `solution/train.py` to forecast `{target_column}` per `{id_column}` and `{date_column}`.

## Runtime contract

The evaluator sets:

- `AUTORESEARCH_TRAIN_DATA`: Parquet training history with columns {train_columns}.
- `AUTORESEARCH_REQUEST`: Parquet with exactly the `{id_column}` and future `{date_column}`
  rows to forecast.
- `AUTORESEARCH_OUTPUT`: destination path.

Write a Parquet file to `AUTORESEARCH_OUTPUT` with exactly `{id_column}`, `{date_column}`,
and `forecast`. Every requested row must appear once. Forecasts must be finite and
non-negative.

Only edit files under `solution/`. You cannot see validation or holdout actuals. Prefer a
focused, testable experiment over broad rewrites.
"""

_SEED_PYPROJECT = """\
[project]
name = "{name}-experiment"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = [
  "numpy>=2.0",
  "pandas>=2.2",
  "pyarrow>=17",
  "statsmodels>=0.14",
  "xgboost>=2.1",
]
"""

_OPENCODE_JSON = """\
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "external_directory": "deny",
    "question": "deny",
    "doom_loop": "allow",
    "bash": "allow",
    "edit": "allow",
    "read": "allow",
    "write": "allow",
    "webfetch": "deny",
    "websearch": "deny"
  }
}
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def ingest_csv(
    data: str | bytes | Path,
    *,
    name: str,
    tasks_root: Path,
    validation_days: int | None = None,
    holdout_days: int | None = None,
    overwrite: bool = False,
) -> IngestResult:
    """Create a complete task directory from a standard-schema sales CSV."""
    frame = _read_csv(data)
    missing = [c for c in REQUIRED_COLUMNS if c not in [str(c).strip() for c in frame.columns]]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    cleaned, _ = _clean(frame)
    if cleaned.empty:
        raise ValueError("no usable rows remain after cleaning")

    dates = sorted(cleaned[DATE_COLUMN].unique())
    distinct = len(dates)
    horizon = _suggest_horizon(distinct)
    val_days = validation_days or horizon
    hold_days = holdout_days or horizon
    if val_days + hold_days >= distinct:
        raise ValueError(
            f"validation({val_days}) + holdout({hold_days}) days must be fewer than the "
            f"{distinct} distinct dates available"
        )

    train_end = distinct - val_days - hold_days
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end : train_end + val_days])
    hold_dates = set(dates[train_end + val_days :])

    train = cleaned[cleaned[DATE_COLUMN].isin(train_dates)]
    validation = cleaned[cleaned[DATE_COLUMN].isin(val_dates)]
    holdout = cleaned[cleaned[DATE_COLUMN].isin(hold_dates)]

    slug = slugify(name)
    task_dir = (tasks_root / slug).resolve()
    if task_dir.exists() and not overwrite:
        raise FileExistsError(f"task '{slug}' already exists at {task_dir}")
    if task_dir.exists():
        import shutil

        shutil.rmtree(task_dir)

    train_path = task_dir / "seed" / "data" / "train.parquet"
    _write(train_path.parent / ".keep", "")
    train.to_parquet(train_path, index=False)
    private = task_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    actual_columns = [ID_COLUMN, DATE_COLUMN, TARGET_COLUMN]
    validation[actual_columns].to_parquet(private / "validation.parquet", index=False)
    holdout[actual_columns].to_parquet(private / "holdout.parquet", index=False)

    _write(
        task_dir / "seed" / "solution" / "train.py",
        _BASELINE_TEMPLATE.format(
            id_column=ID_COLUMN, date_column=DATE_COLUMN, target_column=TARGET_COLUMN
        ),
    )
    _write(
        task_dir / "seed" / "TASK.md",
        _TASK_MD_TEMPLATE.format(
            title=f"{name} demand forecasting",
            id_column=ID_COLUMN,
            date_column=DATE_COLUMN,
            target_column=TARGET_COLUMN,
            train_columns=", ".join(f"`{c}`" for c in cleaned.columns),
        ),
    )
    _write(task_dir / "seed" / "pyproject.toml", _SEED_PYPROJECT.format(name=slug))
    _write(task_dir / "seed" / "opencode.json", _OPENCODE_JSON)
    _write(
        task_dir / "seed" / ".gitignore",
        ".venv/\nforecasts.parquet\n.autoresearch-*-request.parquet\n__pycache__/\n",
    )

    config = {
        "name": slug,
        "description": (
            f"Forecast daily {TARGET_COLUMN} for {int(cleaned[ID_COLUMN].nunique())} items from "
            f"an uploaded sales history ({cleaned.columns.tolist()}). Produce one non-negative "
            f"forecast per requested {ID_COLUMN}/{DATE_COLUMN} pair. A protected evaluator scores "
            f"a {val_days}-day validation window and a later hidden {hold_days}-day holdout."
        ),
        "goal": f"Reduce forecast error on {TARGET_COLUMN} while keeping runtime and holdout stable.",
        "metric": {"name": "wmape", "direction": "min"},
        "secondary_metrics": ["mape", "rmse", "bias_pct", "runtime_s"],
        "guardrails": ["runtime_s<=600"],
        "director": {"model": DEFAULT_MODEL, "temperature": 0.35},
        "agents": {"model": DEFAULT_MODEL, "count": 3, "timeout_s": 1200, "budget_s": 3600},
        "budget": {"max_experiments": 12, "train_timeout_s": 600, "rounds": 4},
        "data": {
            "train": "seed/data/train.parquet",
            "validation_actuals": "private/validation.parquet",
            "holdout_actuals": "private/holdout.parquet",
            "output": "forecasts.parquet",
            "id_column": ID_COLUMN,
            "date_column": DATE_COLUMN,
            "target_column": TARGET_COLUMN,
        },
        "workspace": {
            "seed": "seed",
            "runs": str(Path("..") / ".." / "runs"),
            "allowed_paths": ["solution/"],
        },
        "idea_hints": [
            "Per-item seasonal-naive or exponential smoothing with weekly seasonality",
            "XGBoost with lag, rolling-mean, calendar, and price features",
            "Price-elasticity features from selling_price",
            "Robust ensembles and per-item bias calibration",
        ],
    }
    config_path = task_dir / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    def _last(values: set) -> str:
        return pd.Timestamp(max(values)).strftime("%Y-%m-%d")

    return IngestResult(
        task=slug,
        config_path=config_path,
        task_dir=task_dir,
        train_rows=len(train),
        validation_rows=len(validation),
        holdout_rows=len(holdout),
        skus=int(cleaned[ID_COLUMN].nunique()),
        train_end=_last(train_dates),
        validation_end=_last(val_dates),
        holdout_end=_last(hold_dates),
    )
