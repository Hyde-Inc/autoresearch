from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoresearch.config import DEFAULT_MODEL, TaskConfig, load_config
from autoresearch.discover import read_repo_file, repo_inventory
from autoresearch.prepare import prepare_workspace, write_baseline


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    (repo / "data").mkdir(parents=True)
    (repo / "models").mkdir()
    (repo / "README.md").write_text("# Project\n")
    (repo / "models" / "baseline.py").write_text("print('legacy')\n")
    (repo / ".gitignore").write_text("data/\n")
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    pd.DataFrame(
        {
            "sku_id": ["a"] * 30 + ["b"] * 30,
            "date": list(dates) * 2,
            "units_sold": np.arange(60, dtype=float),
        }
    ).to_parquet(repo / "data" / "sales.parquet", index=False)
    return repo


def test_task_config_is_fully_optional() -> None:
    config = TaskConfig()
    assert config.director.model == DEFAULT_MODEL
    assert config.agents.model == DEFAULT_MODEL
    assert config.budget.rounds == 4


def test_repo_inventory_finds_data_code_and_docs(tmp_path: Path) -> None:
    inventory = repo_inventory(_repo(tmp_path))
    assert [item["path"] for item in inventory["data_files"]] == ["data/sales.parquet"]
    assert "models/baseline.py" in inventory["code_files"]
    assert "README.md" in inventory["docs_and_config"]


def test_read_repo_file_blocks_escapes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert read_repo_file(repo, "README.md").startswith("# Project")
    with pytest.raises(ValueError, match="escapes"):
        read_repo_file(repo, "../outside.txt")


def test_prepare_workspace_builds_protected_task(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    prepared = prepare_workspace(
        repo,
        train_data="data/sales.parquet",
        id_column="sku_id",
        date_column="date",
        target_column="units_sold",
        validation_days=5,
        holdout_days=5,
        overrides={"budget": {"rounds": 2}},
    )
    task_dir = repo / ".autoresearch" / "task"
    assert prepared.task_dir == task_dir
    assert prepared.train_rows == 40
    assert prepared.validation_rows == 10
    assert prepared.holdout_rows == 10

    config = load_config(prepared.config_path)
    assert config.data.id_column == "sku_id"
    assert config.budget.rounds == 2
    assert config.director.model == DEFAULT_MODEL

    seed = task_dir / "seed"
    assert (seed / "models" / "baseline.py").exists()
    assert (seed / "TASK.md").exists()
    assert (seed / "opencode.json").exists()
    # Raw data must not leak into the seed; only the train split may. The project's own
    # gitignore (which ignores data/) is replaced with a controlled one so the train
    # split survives the seed git repo.
    assert not (seed / "data" / "sales.parquet").exists()
    assert "data/" not in (seed / ".gitignore").read_text()
    train = pd.read_parquet(seed / "data" / "train.parquet")
    validation = pd.read_parquet(task_dir / "private" / "validation.parquet")
    assert train["date"].max() < validation["date"].min()


def test_prepare_workspace_rejects_bad_input(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValueError, match="columns not found"):
        prepare_workspace(
            repo,
            train_data="data/sales.parquet",
            id_column="nope",
            date_column="date",
            target_column="units_sold",
            validation_days=5,
            holdout_days=5,
        )
    with pytest.raises(ValueError, match="fewer than"):
        prepare_workspace(
            repo,
            train_data="data/sales.parquet",
            id_column="sku_id",
            date_column="date",
            target_column="units_sold",
            validation_days=20,
            holdout_days=20,
        )


def test_write_baseline_checks_syntax(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    prepared = prepare_workspace(
        repo,
        train_data="data/sales.parquet",
        id_column="sku_id",
        date_column="date",
        target_column="units_sold",
        validation_days=5,
        holdout_days=5,
    )
    config = load_config(prepared.config_path)
    with pytest.raises(ValueError, match="syntax error"):
        write_baseline(config, "def broken(:\n")
    destination = write_baseline(config, "print('ok')\n")
    assert destination.read_text() == "print('ok')\n"
