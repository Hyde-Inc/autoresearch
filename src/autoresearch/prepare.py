"""Build a protected autoresearch task inside any data science repo.

``prepare_workspace`` is called by the Research Director once it has explored
the project and knows where the training data lives. It splits the history
chronologically, copies the project code into a seed workspace (never the raw
data, which would leak validation and holdout actuals), and writes a complete
generated task.yaml under ``<repo>/.autoresearch/task``.

The baseline is intentionally absent: the director writes ``solution/train.py``
itself via ``write_baseline`` - wrapping the project's existing model when
there is one, or bootstrapping a first model from EDA and its skills when
there is not.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .config import TaskConfig
from .discover import DATA_SUFFIXES, SKIP_DIRS
from .eda import load_table
from .ingest import _OPENCODE_JSON, _SEED_PYPROJECT, _TASK_MD_TEMPLATE, slugify

TASK_DIR_NAME = ".autoresearch"


@dataclass
class PreparedTask:
    config_path: Path
    task_dir: Path
    train_rows: int
    validation_rows: int
    holdout_rows: int
    items: int
    train_end: str
    validation_end: str
    holdout_end: str


def _clean(
    frame: pd.DataFrame, id_column: str, date_column: str, target_column: str
) -> pd.DataFrame:
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame[id_column] = frame[id_column].astype(str).str.strip()
    frame = frame.dropna(subset=[id_column, date_column, target_column])
    frame = frame[frame[id_column] != ""]
    frame.loc[frame[target_column] < 0, target_column] = 0.0
    frame = frame[~frame.duplicated([id_column, date_column], keep="last")]
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)


def _copy_project_code(repo: Path, seed: Path) -> None:
    """Copy the project into the seed, excluding raw data (leakage) and clutter."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set()
        for entry in names:
            path = Path(directory) / entry
            if entry in SKIP_DIRS or entry == ".gitignore" or entry.startswith("."):
                skipped.add(entry)
            elif path.is_file() and path.suffix.lower() in DATA_SUFFIXES:
                skipped.add(entry)
        return skipped

    shutil.copytree(repo, seed, ignore=ignore)


def prepare_workspace(
    repo: Path,
    *,
    train_data: str,
    id_column: str,
    date_column: str,
    target_column: str,
    validation_days: int,
    holdout_days: int,
    name: str | None = None,
    overrides: dict | None = None,
) -> PreparedTask:
    repo = repo.resolve()
    data_path = (repo / train_data).resolve()
    if repo not in data_path.parents:
        raise ValueError(f"training data must live inside the project: {train_data}")
    frame = load_table(data_path)
    missing = [c for c in (id_column, date_column, target_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not found in {train_data}: {', '.join(missing)}")
    cleaned = _clean(frame, id_column, date_column, target_column)
    if cleaned.empty:
        raise ValueError("no usable rows remain after cleaning")

    dates = sorted(cleaned[date_column].unique())
    if validation_days + holdout_days >= len(dates):
        raise ValueError(
            f"validation({validation_days}) + holdout({holdout_days}) days must be fewer "
            f"than the {len(dates)} distinct dates available"
        )
    train_end = len(dates) - validation_days - holdout_days
    train = cleaned[cleaned[date_column].isin(dates[:train_end])]
    validation = cleaned[cleaned[date_column].isin(dates[train_end : train_end + validation_days])]
    holdout = cleaned[cleaned[date_column].isin(dates[train_end + validation_days :])]

    task_dir = repo / TASK_DIR_NAME / "task"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    seed = task_dir / "seed"
    _copy_project_code(repo, seed)
    (seed / "data").mkdir(parents=True, exist_ok=True)
    train.to_parquet(seed / "data" / "train.parquet", index=False)
    private = task_dir / "private"
    private.mkdir()
    actual_columns = [id_column, date_column, target_column]
    validation[actual_columns].to_parquet(private / "validation.parquet", index=False)
    holdout[actual_columns].to_parquet(private / "holdout.parquet", index=False)

    slug = slugify(name or repo.name)
    (seed / "TASK.md").write_text(
        _TASK_MD_TEMPLATE.format(
            title=f"{slug} forecasting",
            id_column=id_column,
            date_column=date_column,
            target_column=target_column,
            train_columns=", ".join(f"`{c}`" for c in cleaned.columns),
        )
    )
    if not (seed / "pyproject.toml").exists():
        (seed / "pyproject.toml").write_text(_SEED_PYPROJECT.format(name=slug))
    (seed / "opencode.json").write_text(_OPENCODE_JSON)
    (seed / ".gitignore").write_text(
        ".venv/\nforecasts.parquet\n.autoresearch-*-request.parquet\n__pycache__/\n"
    )
    (seed / "solution").mkdir(exist_ok=True)

    overlay = overrides or {}
    config = {
        "name": overlay.get("name", slug),
        "description": (
            f"Forecast {target_column} per {id_column} and {date_column} for the "
            f"{repo.name} project. A protected evaluator scores a {validation_days}-day "
            f"validation window and a later hidden {holdout_days}-day holdout."
        ),
        "goal": "",
        "metric": overlay.get("metric", {"name": "wmape", "direction": "min"}),
        "secondary_metrics": ["mape", "rmse", "bias_pct", "runtime_s"],
        "guardrails": overlay.get("guardrails", ["runtime_s<=600"]),
        "director": overlay.get("director", {}),
        "agents": overlay.get("agents", {}),
        "budget": overlay.get("budget", {}),
        "data": {
            "train": "seed/data/train.parquet",
            "validation_actuals": "private/validation.parquet",
            "holdout_actuals": "private/holdout.parquet",
            "output": "forecasts.parquet",
            "id_column": id_column,
            "date_column": date_column,
            "target_column": target_column,
        },
        "workspace": {"seed": "seed", "runs": "../runs", "allowed_paths": ["solution/"]},
    }
    if "skills" in overlay:
        config["skills"] = [
            str(Path("..") / ".." / skill) for skill in overlay["skills"]
        ]
    config_path = task_dir / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    def _last(split: pd.DataFrame) -> str:
        return pd.Timestamp(split[date_column].max()).strftime("%Y-%m-%d")

    return PreparedTask(
        config_path=config_path,
        task_dir=task_dir,
        train_rows=len(train),
        validation_rows=len(validation),
        holdout_rows=len(holdout),
        items=int(cleaned[id_column].nunique()),
        train_end=_last(train),
        validation_end=_last(validation),
        holdout_end=_last(holdout),
    )


def write_baseline(config: TaskConfig, code: str) -> Path:
    """Write the director-authored baseline into the seed, checking syntax first."""
    try:
        compile(code, "train.py", "exec")
    except SyntaxError as exc:
        raise ValueError(f"baseline code has a syntax error: {exc}") from exc
    destination = config.resolve(config.workspace.seed) / "solution" / "train.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(code)
    return destination
