"""Tests for the Foundry runtime: scoring, build gating, and worker routing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoresearch import foundry, harness_foundry
from autoresearch.config import TaskConfig
from autoresearch.foundry_setup import read_gradle_properties, scaffold
from autoresearch.harness_foundry import _evaluate_sync, _score_split, _start_build
from autoresearch.orchestrator import _main_ref
from autoresearch.worker import _entry_path, _prompt_rules, _task_doc, _verify_build

TRANSFORM_REL = "transforms-python/src/myproject/datasets/forecast.py"


def foundry_config(tmp_path: Path) -> TaskConfig:
    config = TaskConfig.model_validate(
        {
            "runtime": "foundry",
            "data": {"id_column": "sku_id", "date_column": "date", "target_column": "units_sold"},
            "workspace": {"allowed_paths": [TRANSFORM_REL]},
            "foundry": {
                "repo_dir": str(tmp_path),
                "branch": "master",
                "datasets": {
                    "sales_train": "ri.t",
                    "forecast_request": "ri.r",
                    "forecasts": "ri.f",
                    "validation_actuals": "ri.v",
                    "holdout_actuals": "ri.h",
                },
            },
        }
    )
    config.config_path = tmp_path / "task.yaml"
    return config


def frame(rows: list[tuple[str, str, float]], value_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            value_column: [r[2] for r in rows],
        }
    )


class TestScoreSplit:
    def test_scores_subset_of_forecasts(self, tmp_path: Path) -> None:
        # Forecasts cover validation + holdout; actuals are just validation.
        forecasts = frame(
            [("A", "2024-01-01", 10.0), ("A", "2024-01-02", 20.0), ("A", "2024-01-03", 30.0)],
            "forecast",
        )
        actuals = frame([("A", "2024-01-01", 8.0), ("A", "2024-01-02", 22.0)], "units_sold")
        metrics, merged = _score_split(forecasts, actuals, foundry_config(tmp_path))
        assert len(merged) == 2
        assert metrics["wmape"] == pytest.approx(4.0 / 30.0)

    def test_missing_rows_fail(self, tmp_path: Path) -> None:
        forecasts = frame([("A", "2024-01-01", 10.0)], "forecast")
        actuals = frame([("A", "2024-01-01", 8.0), ("A", "2024-01-02", 22.0)], "units_sold")
        with pytest.raises(ValueError, match="missing 1 of 2"):
            _score_split(forecasts, actuals, foundry_config(tmp_path))

    def test_duplicates_fail(self, tmp_path: Path) -> None:
        forecasts = frame(
            [("A", "2024-01-01", 10.0), ("A", "2024-01-01", 11.0)], "forecast"
        )
        actuals = frame([("A", "2024-01-01", 8.0)], "units_sold")
        with pytest.raises(ValueError, match="duplicate"):
            _score_split(forecasts, actuals, foundry_config(tmp_path))

    def test_negative_forecasts_fail(self, tmp_path: Path) -> None:
        forecasts = frame([("A", "2024-01-01", -1.0)], "forecast")
        actuals = frame([("A", "2024-01-01", 8.0)], "units_sold")
        with pytest.raises(ValueError, match="non-negative"):
            _score_split(forecasts, actuals, foundry_config(tmp_path))

    def test_timezone_aware_dates_align(self, tmp_path: Path) -> None:
        forecasts = frame([("A", "2024-01-01", 10.0)], "forecast")
        forecasts["date"] = forecasts["date"].dt.tz_localize("UTC")
        actuals = frame([("A", "2024-01-01", 10.0)], "units_sold")
        metrics, _ = _score_split(forecasts, actuals, foundry_config(tmp_path))
        assert metrics["wmape"] == 0.0


class TestStartBuild:
    def test_up_to_date_without_new_code_skips_build(self, tmp_path, monkeypatch) -> None:
        def raise_up_to_date(*args, **kwargs):
            raise foundry.FoundryError("Orchestration:BuildTargetsUpToDate")

        monkeypatch.setattr(harness_foundry, "create_build", raise_up_to_date)
        rid = _start_build(
            foundry_config(tmp_path), "master", expect_new_code=False, on_progress=None
        )
        assert rid is None

    def test_waits_for_publish_when_new_code(self, tmp_path, monkeypatch) -> None:
        calls = {"n": 0}

        def pending_then_ready(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise foundry.FoundryError("Orchestration:BuildTargetsUpToDate")
            return "ri.build.1"

        monkeypatch.setattr(harness_foundry, "create_build", pending_then_ready)
        monkeypatch.setattr(harness_foundry.time, "sleep", lambda _s: None)
        rid = _start_build(
            foundry_config(tmp_path), "exp-branch", expect_new_code=True, on_progress=None
        )
        assert rid == "ri.build.1"
        assert calls["n"] == 3

    def test_other_errors_raise(self, tmp_path, monkeypatch) -> None:
        def raise_other(*args, **kwargs):
            raise foundry.FoundryError("PermissionDenied")

        monkeypatch.setattr(harness_foundry, "create_build", raise_other)
        with pytest.raises(foundry.FoundryError, match="PermissionDenied"):
            _start_build(foundry_config(tmp_path), "master", expect_new_code=True, on_progress=None)


class TestEvaluateSync:
    def test_validation_and_holdout_from_one_build(self, tmp_path, monkeypatch) -> None:
        forecasts = frame(
            [("A", "2024-01-01", 10.0), ("A", "2024-02-01", 30.0)], "forecast"
        )
        validation = frame([("A", "2024-01-01", 10.0)], "units_sold")
        holdout = frame([("A", "2024-02-01", 20.0)], "units_sold")
        reads: list[str] = []

        def fake_train(worktree, config, branch, *, push, on_progress):
            assert push is False
            return forecasts, 12.0

        def fake_read(rid, *, branch=None, **kwargs):
            reads.append(rid)
            return {"ri.v": validation, "ri.h": holdout}[rid]

        monkeypatch.setattr(harness_foundry, "_train_on_foundry", fake_train)
        monkeypatch.setattr(harness_foundry.foundry, "read_dataset", fake_read)
        config = foundry_config(tmp_path)
        result = _evaluate_sync(
            tmp_path,
            config,
            {"wmape": 0.9, "holdout_wmape": 0.2},
            None,
            None,
            include_holdout=True,
            push=False,
            branch="master",
        )
        assert result.error is None
        assert result.metrics["wmape"] == 0.0
        assert result.holdout_metrics["wmape"] == pytest.approx(0.5)
        # Holdout worse than incumbent 0.2 -> final gate fails, but metrics kept.
        assert not result.passed
        assert any("holdout" in failure for failure in result.guardrail_failures)
        assert reads == ["ri.v", "ri.h"]

    def test_errors_become_scored_failures(self, tmp_path, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise foundry.FoundryError("build exploded")

        monkeypatch.setattr(harness_foundry, "_train_on_foundry", boom)
        result = _evaluate_sync(
            tmp_path,
            foundry_config(tmp_path),
            None,
            None,
            None,
            include_holdout=False,
            push=False,
            branch="master",
        )
        assert result.error == "build exploded"
        assert not result.passed


class TestWorkerRouting:
    def test_entry_point_and_docs_per_runtime(self, tmp_path: Path) -> None:
        config = foundry_config(tmp_path)
        assert _entry_path(config) == TRANSFORM_REL
        assert _task_doc(config) == "AUTORESEARCH.md"
        assert TRANSFORM_REL in _prompt_rules(config)
        local = TaskConfig()
        assert _entry_path(local) == "solution/train.py"
        assert _task_doc(local) == "TASK.md"

    def test_verify_build_checks_the_transform(self, tmp_path: Path) -> None:
        config = foundry_config(tmp_path)
        assert "does not exist" in _verify_build(tmp_path, config)
        target = tmp_path / TRANSFORM_REL
        target.parent.mkdir(parents=True)
        target.write_text("def build_forecasts(train, request):\n    return request\n")
        assert _verify_build(tmp_path, config) is None
        target.write_text("def broken(:\n")
        assert "syntax error" in _verify_build(tmp_path, config)

    def test_main_ref_per_runtime(self, tmp_path: Path) -> None:
        assert _main_ref(TaskConfig()) == "main"
        assert _main_ref(foundry_config(tmp_path)) == "master"


class TestScaffold:
    def _make_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        (repo / "transforms-python" / "src" / "acme_project").mkdir(parents=True)
        (repo / "transforms-python" / "conda_recipe").mkdir(parents=True)
        (repo / "gradle.properties").write_text(
            "transformsRepoRid = ri.stemma.main.repository.abc\n"
            "transformsRepoPath = /org/Some Project/Code repository - x\n"
            "transformsDefaultBranchName = master\n"
        )
        (repo / "transforms-python" / "conda_recipe" / "meta.yaml").write_text(
            "requirements:\n  build:\n    - python\n  run:\n    - python\n"
            "    - transforms {{ PYTHON_TRANSFORMS_VERSION }}\n\nbuild:\n  noarch: python\n"
        )
        return repo

    def test_gradle_properties_parse(self, tmp_path: Path) -> None:
        repo = self._make_repo(tmp_path)
        props = read_gradle_properties(repo)
        assert props["transformsRepoRid"] == "ri.stemma.main.repository.abc"
        assert props["transformsRepoPath"].startswith("/org/Some Project")

    def test_scaffold_writes_transform_contract_and_deps(self, tmp_path: Path) -> None:
        repo = self._make_repo(tmp_path)
        rel, _actions = scaffold(repo, "/org/Some Project/autoresearch", "master")
        assert rel == "transforms-python/src/acme_project/datasets/forecast.py"
        transform = (repo / rel).read_text()
        assert '"/org/Some Project/autoresearch/sales_train"' in transform
        assert "def build_forecasts" in transform
        contract = (repo / "AUTORESEARCH.md").read_text()
        assert rel in contract
        recipe = (repo / "transforms-python" / "conda_recipe" / "meta.yaml").read_text()
        for pkg in ("scikit-learn", "xgboost", "lightgbm", "statsmodels"):
            assert f"- {pkg}" in recipe
        assert recipe.index("- xgboost") < recipe.index("\nbuild:")

    def test_scaffold_never_clobbers_an_existing_transform(self, tmp_path: Path) -> None:
        repo = self._make_repo(tmp_path)
        rel, _ = scaffold(repo, "/org/Some Project/autoresearch", "master")
        (repo / rel).write_text("# improved model\n")
        _, actions = scaffold(repo, "/org/Some Project/autoresearch", "master")
        assert (repo / rel).read_text() == "# improved model\n"
        assert not any("baseline transform" in action for action in actions)
