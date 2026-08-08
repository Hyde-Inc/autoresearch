import io
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.console import Console

from autoresearch.config import load_config
from autoresearch.interview import ResearchBrief, SetupSession, apply_brief, data_profile


def _task(tmp_path: Path) -> Path:
    seed = tmp_path / "seed"
    (seed / "data").mkdir(parents=True)
    (seed / "solution").mkdir(parents=True)
    (seed / "solution" / "train.py").write_text("print('baseline')\n")
    pd.DataFrame(
        {
            "sku": ["a", "a", "b", "b"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
            "demand": [0, 2, 3, 4],
        }
    ).to_parquet(seed / "data" / "train.parquet", index=False)
    config = {
        "name": "test",
        "description": "Forecast demand.",
        "goal": "Lower WMAPE.",
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    (repo / "data").mkdir(parents=True)
    (repo / "models").mkdir()
    (repo / "README.md").write_text("# Demand project\nData lives in data/.\n")
    (repo / "models" / "baseline.py").write_text("print('legacy model')\n")
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    frame = pd.concat(
        pd.DataFrame(
            {
                "sku_id": item,
                "date": dates,
                "units_sold": np.arange(len(dates), dtype=float) % 7 + 1,
            }
        )
        for item in ("a", "b")
    )
    frame.to_parquet(repo / "data" / "sales.parquet", index=False)
    return repo


def _console() -> Console:
    return Console(file=io.StringIO(), width=100)


def _session(tmp_path: Path) -> SetupSession:
    return SetupSession(object(), _console(), 2, config_path=_task(tmp_path))


def test_data_profile_and_apply_brief(tmp_path: Path) -> None:
    config = load_config(_task(tmp_path))
    profile = data_profile(config)
    assert profile["items"] == 2
    assert profile["zero_share"] == 0.25
    apply_brief(
        config,
        ResearchBrief(
            goal="Reduce WMAPE by 10%.",
            context="Stockouts cost more than overstock.",
            guardrails=["runtime_s<=60"],
            idea_hints=["Try global tree models"],
        ),
    )
    updated = load_config(config.config_path)
    assert updated.goal == "Reduce WMAPE by 10%."
    assert updated.context.startswith("Stockouts")
    assert updated.guardrails == ["runtime_s<=60"]


def test_finalize_brief_tool_saves_task_yaml(tmp_path: Path) -> None:
    session = _session(tmp_path)
    result = session.execute(
        "finalize_brief",
        {
            "goal": "Cut WMAPE",
            "context": "Symmetric costs.",
            "guardrails": ["runtime_s<=600"],
            "idea_hints": ["xgboost"],
            "metric_name": "wmape",
        },
    )
    assert result["ok"] is True
    assert session.config.goal == "Cut WMAPE"


def test_replace_baseline_tool_returns_error_instead_of_crashing(tmp_path: Path) -> None:
    session = _session(tmp_path)
    result = session.execute("replace_baseline", {"path": "y"})
    assert "baseline script not found" in result["error"]


def test_start_research_tool_parses_ideas_and_filters_skills(tmp_path: Path) -> None:
    session = _session(tmp_path)
    result = session.execute(
        "start_research",
        {
            "ideas": [
                {
                    "title": "Global XGBoost",
                    "hypothesis": "Trees beat naive.",
                    "instructions": "Build lag features.",
                    "category": "ml",
                    "skills_used": ["tree-model-features", "made-up-skill"],
                },
                {
                    "title": "Bias calibration",
                    "hypothesis": "Calibration trims bias.",
                    "instructions": "Fit residual correction.",
                },
                {
                    "title": "Extra idea beyond budget",
                    "hypothesis": "Should be trimmed.",
                    "instructions": "n/a",
                },
            ]
        },
    )
    assert result == {"ok": True, "experiments": 2}
    assert session.final_ideas is not None
    assert [idea.title for idea in session.final_ideas] == ["Global XGBoost", "Bias calibration"]
    assert session.final_ideas[0].skills_used == ["tree-model-features"]


def test_unknown_tool_and_missing_pending_metric(tmp_path: Path) -> None:
    session = _session(tmp_path)
    assert "unknown tool" in session.execute("does_not_exist", {})["error"]
    assert "no custom metric" in session.execute("adopt_custom_metric", {})["error"]


def test_repo_session_explores_project_itself(tmp_path: Path) -> None:
    session = SetupSession(object(), _console(), 2, repo=_repo(tmp_path))
    prompt = session.system_prompt()
    assert "data/sales.parquet" in prompt
    assert "models/baseline.py" in prompt

    read = session.execute("read_file", {"path": "README.md"})
    assert "Demand project" in read["content"]
    assert "escapes the project" in session.execute("read_file", {"path": "../secrets"})["error"]

    profile = session.execute("explore_data", {"path": "data/sales.parquet", "analysis": "profile"})
    assert profile["rows"] == 80
    seasonality = session.execute(
        "explore_data",
        {
            "path": "data/sales.parquet",
            "analysis": "seasonality",
            "date_column": "date",
            "target_column": "units_sold",
        },
    )
    assert seasonality["daily_total_autocorrelation"]["lag_7"] > 0.9


def test_repo_session_requires_workspace_before_brief(tmp_path: Path) -> None:
    session = SetupSession(object(), _console(), 1, repo=_repo(tmp_path))
    result = session.execute(
        "finalize_brief",
        {"goal": "g", "context": "c", "guardrails": [], "idea_hints": [], "metric_name": "wmape"},
    )
    assert "prepare_workspace" in result["error"]


def test_repo_session_prepares_workspace_and_writes_baseline(tmp_path: Path) -> None:
    session = SetupSession(object(), _console(), 1, repo=_repo(tmp_path))
    prepared = session.execute(
        "prepare_workspace",
        {
            "train_data": "data/sales.parquet",
            "id_column": "sku_id",
            "date_column": "date",
            "target_column": "units_sold",
            "validation_days": 7,
            "holdout_days": 7,
        },
    )
    assert prepared["ok"] is True
    assert session.config is not None
    assert session.config.data.id_column == "sku_id"

    bad = session.execute("write_baseline", {"code": "def broken(:\n"})
    assert "syntax error" in bad["error"]
    good = session.execute("write_baseline", {"code": "print('model')\n"})
    assert good["ok"] is True
    written = Path(good["written_to"])
    assert written.read_text() == "print('model')\n"
    assert written == session.config.resolve(session.config.workspace.seed) / "solution/train.py"
