"""Tests for the Foundry runtime: scoring, build gating, and worker routing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoresearch import foundry, harness_foundry
from autoresearch.config import TaskConfig
from autoresearch.foundry_setup import read_gradle_properties, scaffold
from autoresearch.harness_foundry import (
    _ensure_output_branch,
    _evaluate_sync,
    _score_split,
    _start_build,
)
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


class TestBuildTimeout:
    def test_overrunning_build_is_cancelled_on_foundry(self, monkeypatch) -> None:
        from autoresearch import foundry_build

        cancelled: list[str] = []
        monkeypatch.setattr(
            foundry_build, "get_build", lambda rid: {"status": "RUNNING", "jobRids": []}
        )
        monkeypatch.setattr(
            foundry_build, "cancel_build", lambda rid: cancelled.append(rid)
        )
        monkeypatch.setattr(foundry_build.time, "sleep", lambda _s: None)
        with pytest.raises(foundry_build.FoundryBuildError, match="exceeded the 0s timeout"):
            foundry_build.wait_for_build("ri.build.slow", timeout_s=0, poll_s=1)
        assert cancelled == ["ri.build.slow"]

    def test_cancel_failure_still_raises_timeout(self, monkeypatch) -> None:
        from autoresearch import foundry_build

        def cancel_fails(rid: str) -> None:
            raise foundry.FoundryError("no permission to cancel")

        monkeypatch.setattr(
            foundry_build, "get_build", lambda rid: {"status": "RUNNING", "jobRids": []}
        )
        monkeypatch.setattr(foundry_build, "cancel_build", cancel_fails)
        monkeypatch.setattr(foundry_build.time, "sleep", lambda _s: None)
        with pytest.raises(foundry_build.FoundryBuildError, match="cancel request failed"):
            foundry_build.wait_for_build("ri.build.slow", timeout_s=0, poll_s=1)


class TestEnsureOutputBranch:
    def test_seeds_missing_branch_from_main_transaction(self, tmp_path, monkeypatch) -> None:
        # A fresh branch must be seeded with master's transaction so Foundry
        # reports UpToDate (not "output missing") until the branch publishes -
        # otherwise the first build runs the fallback (baseline) job spec.
        created: list[tuple] = []
        branches = {"master": {"name": "master", "transactionRid": "ri.txn.1"}}
        monkeypatch.setattr(
            harness_foundry.foundry, "get_branch", lambda rid, name: branches.get(name)
        )
        monkeypatch.setattr(
            harness_foundry.foundry,
            "create_branch",
            lambda rid, name, transaction_rid=None: created.append((rid, name, transaction_rid)),
        )
        _ensure_output_branch(foundry_config(tmp_path), "autoresearch/exp-1")
        assert created == [("ri.f", "autoresearch/exp-1", "ri.txn.1")]

    def test_existing_branch_untouched(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            harness_foundry.foundry,
            "get_branch",
            lambda rid, name: {"name": name, "transactionRid": "ri.txn.2"},
        )
        monkeypatch.setattr(
            harness_foundry.foundry,
            "create_branch",
            lambda *a, **k: pytest.fail("must not create an existing branch"),
        )
        _ensure_output_branch(foundry_config(tmp_path), "autoresearch/exp-1")


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

    def test_scaffold_writes_pipeline_contract_and_deps(self, tmp_path: Path) -> None:
        from autoresearch.foundry_setup import default_refs

        repo = self._make_repo(tmp_path)
        rel, _actions = scaffold(repo, default_refs("/org/Some Project/autoresearch"), "master")
        datasets = repo / "transforms-python" / "src" / "acme_project" / "datasets"
        assert rel == "transforms-python/src/acme_project/datasets/model_running/forecast.py"
        # Three pipeline stages, each an importable package.
        for stage, filename in (
            ("data_preprocessing", "preprocess.py"),
            ("model_running", "forecast.py"),
            ("model_evaluation", "evaluate.py"),
        ):
            assert (datasets / stage / "__init__.py").exists()
            assert (datasets / stage / filename).exists()
        preprocess = (datasets / "data_preprocessing" / "preprocess.py").read_text()
        assert '"/org/Some Project/autoresearch/sales_raw"' in preprocess
        assert "def clean" in preprocess
        transform = (repo / rel).read_text()
        assert '"/org/Some Project/autoresearch/sales_train"' in transform
        assert "def build_forecasts" in transform
        evaluate = (datasets / "model_evaluation" / "evaluate.py").read_text()
        assert '"/org/Some Project/autoresearch/evaluation_metrics"' in evaluate
        contract = (repo / "AUTORESEARCH.md").read_text()
        assert rel in contract
        recipe = (repo / "transforms-python" / "conda_recipe" / "meta.yaml").read_text()
        for pkg in ("scikit-learn", "xgboost", "lightgbm", "statsmodels"):
            assert f"- {pkg}" in recipe
        assert recipe.index("- xgboost") < recipe.index("\nbuild:")

    def test_scaffold_never_clobbers_an_existing_transform(self, tmp_path: Path) -> None:
        from autoresearch.foundry_setup import default_refs

        repo = self._make_repo(tmp_path)
        refs = default_refs("/org/Some Project/autoresearch")
        rel, _ = scaffold(repo, refs, "master")
        (repo / rel).write_text("# improved model\n")
        _, actions = scaffold(repo, refs, "master")
        assert (repo / rel).read_text() == "# improved model\n"
        assert not any("forecast.py" in action for action in actions)

    def test_scaffold_migrates_legacy_single_file_layout(self, tmp_path: Path) -> None:
        from autoresearch.foundry_setup import default_refs

        repo = self._make_repo(tmp_path)
        datasets = repo / "transforms-python" / "src" / "acme_project" / "datasets"
        datasets.mkdir(parents=True)
        legacy = datasets / "forecast.py"
        legacy.write_text("# agent-improved winning model\n")
        rel, actions = scaffold(repo, default_refs("/org/Some Project/autoresearch"), "master")
        assert (repo / rel).read_text() == "# agent-improved winning model\n"
        assert not legacy.exists()
        assert any("migrated" in action for action in actions)

    def test_scaffold_existing_datasets_by_rid_with_custom_columns(self, tmp_path: Path) -> None:
        repo = self._make_repo(tmp_path)
        refs = {
            "sales_train": "ri.foundry.main.dataset.train",
            "forecast_request": "ri.foundry.main.dataset.request",
            "forecasts": "ri.foundry.main.dataset.fcst",
            "validation_actuals": "ri.foundry.main.dataset.val",
            "holdout_actuals": "ri.foundry.main.dataset.hold",
            "evaluation_metrics": "ri.foundry.main.dataset.eval",
            # no sales_raw: the data is already clean upstream
        }
        rel, _ = scaffold(
            repo, refs, "master", id_col="item_id", date_col="ds", target_col="demand"
        )
        datasets = repo / "transforms-python" / "src" / "acme_project" / "datasets"
        # No raw feed wired -> no preprocessing stage.
        assert not (datasets / "data_preprocessing").exists()
        transform = (repo / rel).read_text()
        assert 'SALES_TRAIN = "ri.foundry.main.dataset.train"' in transform
        assert 'ID = "item_id"' in transform
        assert 'DATE = "ds"' in transform
        assert 'TARGET = "demand"' in transform
        evaluate = (datasets / "model_evaluation" / "evaluate.py").read_text()
        assert 'EVALUATION_METRICS = "ri.foundry.main.dataset.eval"' in evaluate
        contract = (repo / "AUTORESEARCH.md").read_text()
        assert "`item_id` (str), `ds` (date), `demand` (float)" in contract

    def test_pipeline_targets_skip_sales_train_without_raw_feed(self) -> None:
        from autoresearch.config import FoundryDatasets

        provisioned = FoundryDatasets(
            sales_raw="ri.raw", sales_train="ri.train", forecasts="ri.fcst",
            evaluation_metrics="ri.eval",
        )
        assert provisioned.pipeline_targets() == ["ri.train", "ri.fcst", "ri.eval"]
        existing = FoundryDatasets(sales_train="ri.train", forecasts="ri.fcst")
        # No sales_raw -> sales_train is data, not a transform output.
        assert existing.pipeline_targets() == ["ri.fcst"]

    def test_make_raw_injects_quality_issues(self) -> None:
        from autoresearch.foundry_setup import generate_history, make_raw

        clean = generate_history(n_skus=4, n_days=90)
        raw = make_raw(clean)
        assert raw["units_sold"].isna().sum() > 0
        assert (raw["units_sold"].dropna() < 0).sum() > 0
        assert raw.duplicated(["sku_id", "date"]).sum() > 0
        assert len(raw) < len(clean) * 1.02  # some rows dropped, some duplicated
