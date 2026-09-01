import io
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rich.console import Console

from autoresearch.config import load_config
from autoresearch.interview import ResearchBrief, SetupSession, apply_brief, data_profile
from autoresearch.slash import SessionSettings


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
    return SetupSession(
        object(), _console(), settings=SessionSettings(n_agents=2), config_path=_task(tmp_path)
    )


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


_PLAN_IDEAS = {
    "ideas": [
        {
            "title": "Global XGBoost",
            "hypothesis": "Trees beat naive.",
            "instructions": "Build lag features.",
            "category": "ml",
            "skills_used": ["boosting-demand-models", "made-up-skill"],
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
}


def test_write_plan_tool_writes_editable_file_without_launching(tmp_path: Path) -> None:
    session = _session(tmp_path)
    result = session.execute("write_plan", _PLAN_IDEAS)
    assert result["ok"] is True
    assert result["experiments"] == 2  # trimmed to n_agents
    assert session.final_ideas is None  # nothing launches until the user says execute
    assert session.plan_path is not None and session.plan_path.exists()
    text = session.plan_path.read_text()
    assert "## Experiment 1: Global XGBoost" in text
    assert "made-up-skill" not in text  # unknown skills filtered
    # The plan lives in a numbered, goal-named session folder under research/.
    assert session.plan_path.name == "round-1-plan.md"
    assert session.session_dir == session.plan_path.parent
    assert session.session_dir.parent == tmp_path / "research"
    assert session.session_dir.name == "001-lower-wmape"
    assert (session.session_dir / "README.md").exists()
    # A second write_plan call (revision) reuses the same file and folder.
    session.execute("write_plan", _PLAN_IDEAS)
    assert len(list(session.session_dir.glob("round-*-plan.md"))) == 1


def test_execute_gate_runs_the_hand_edited_plan(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path)
    session.execute("write_plan", _PLAN_IDEAS)
    # Human deletes the second experiment and edits the goal before executing.
    text = session.plan_path.read_text()
    text = text[: text.index("## Experiment 2:")]
    text = text.replace("goal: Lower WMAPE.", "goal: cut wmape by 10%")
    session.plan_path.write_text(text)

    inputs = iter(["execute"])
    monkeypatch.setattr("autoresearch.interview.input_box", lambda console, **_: next(inputs))
    assert session._gate_user_message() is None
    assert [idea.title for idea in session.final_ideas] == ["Global XGBoost"]
    assert session.settings.goal == "cut wmape by 10%"
    assert load_config(session.config_path).goal == "cut wmape by 10%"
    assert "status: executed" in session.plan_path.read_text()


def test_gate_passes_feedback_through_and_recovers_from_broken_plans(
    tmp_path: Path, monkeypatch
) -> None:
    session = _session(tmp_path)
    session.execute("write_plan", _PLAN_IDEAS)

    inputs = iter(["make experiment 2 about promotions instead"])
    monkeypatch.setattr("autoresearch.interview.input_box", lambda console, **_: next(inputs))
    assert session._gate_user_message() == "make experiment 2 about promotions instead"
    assert session.final_ideas is None

    # A broken plan reports the problem and keeps asking instead of crashing.
    session.plan_path.write_text("---\ngoal: x\n---\n\nno experiments\n")
    inputs = iter(["execute", "stop"])
    monkeypatch.setattr("autoresearch.interview.input_box", lambda console, **_: next(inputs))
    assert session._gate_user_message() is None
    assert session.final_ideas == []


def test_unknown_tool_and_missing_pending_metric(tmp_path: Path) -> None:
    session = _session(tmp_path)
    assert "unknown tool" in session.execute("does_not_exist", {})["error"]
    assert "no custom metric" in session.execute("adopt_custom_metric", {})["error"]


def test_repo_session_explores_project_itself(tmp_path: Path) -> None:
    session = SetupSession(object(), _console(), repo=_repo(tmp_path))
    prompt = session.system_prompt()
    assert "data/sales.parquet" in prompt
    assert "models/baseline.py" in prompt

    read = session.execute("read_file", {"path": "README.md"})
    assert "Demand project" in read["content"]
    assert "escapes the project" in session.execute("read_file", {"path": "../secrets"})["error"]

    profile = session.execute("explore_data", {"path": "data/sales.parquet", "analysis": "profile"})
    assert profile["analysis_id"] == "A1"
    assert profile["result"]["rows"] == 80
    seasonality = session.execute(
        "explore_data",
        {
            "path": "data/sales.parquet",
            "analysis": "seasonality",
            "date_column": "date",
            "target_column": "units_sold",
        },
    )
    assert seasonality["analysis_id"] == "A2"
    assert seasonality["result"]["daily_total_autocorrelation"]["lag_7"] > 0.9
    # Each run is logged verbatim so round-1 ideas can cite it as evidence.
    assert [record.id for record in session.analysis_log] == ["A1", "A2"]
    assert session.analysis_log[0].tool == "explore_data"


def test_survey_runs_in_background_and_folds_into_prompt(tmp_path: Path, monkeypatch) -> None:
    release = threading.Event()

    def slow_survey(repo, model):
        release.wait(5)
        return "# Repository survey\nmodels/baseline.py is the incumbent"

    monkeypatch.setattr("autoresearch.interview.survey_repo", slow_survey)
    session = SetupSession(object(), _console(), repo=_repo(tmp_path))
    session.start_survey()
    # The chat is usable immediately: survey pending, prompt says to work without it.
    assert session.survey_pending
    assert session.poll_survey() is False
    prompt = session.system_prompt()
    assert "surveying the repository in the background" in prompt
    assert "Project inventory" in prompt
    # When the agent finishes, one poll folds the report into the prompt.
    release.set()
    session._survey_thread.join(timeout=5)
    assert session.poll_survey() is True
    assert not session.survey_pending
    assert "models/baseline.py is the incumbent" in session.system_prompt()


def test_survey_failure_falls_back_to_inventory(tmp_path: Path, monkeypatch) -> None:
    def boom(repo, model):
        raise RuntimeError("opencode exploded")

    monkeypatch.setattr("autoresearch.interview.survey_repo", boom)
    session = SetupSession(object(), _console(), repo=_repo(tmp_path))
    session.start_survey()
    session._survey_thread.join(timeout=5)
    assert session.poll_survey() is False
    assert session.survey is None
    assert "Project inventory" in session.system_prompt()


def test_slash_commands_update_settings_and_prepend_notes(tmp_path: Path, monkeypatch) -> None:
    session = SetupSession(object(), _console(), repo=_repo(tmp_path))
    inputs = iter(["/goal reduce wmape", "/baseline models/baseline.py", "/n_agents 2", "go"])
    monkeypatch.setattr("autoresearch.interview.input_box", lambda console, **_: next(inputs))
    message = session.next_user_message()
    assert session.settings.goal == "reduce wmape"
    assert session.settings.baseline_path == "models/baseline.py"
    assert session.settings.n_agents == 2
    assert message.startswith("[settings updated]")
    assert "baseline model pinned to models/baseline.py" in message
    assert message.endswith("go")


def test_repo_session_requires_workspace_before_brief(tmp_path: Path) -> None:
    session = SetupSession(object(), _console(), repo=_repo(tmp_path))
    result = session.execute(
        "finalize_brief",
        {"goal": "g", "context": "c", "guardrails": [], "idea_hints": [], "metric_name": "wmape"},
    )
    assert "prepare_workspace" in result["error"]


def test_repo_session_prepares_workspace_and_writes_baseline(tmp_path: Path) -> None:
    settings = SessionSettings(n_agents=2, rounds=3, timeout_s=300)
    settings.guardrails.append("bias_pct within -8..8")
    session = SetupSession(object(), _console(), settings=settings, repo=_repo(tmp_path))
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
    # Slash settings flow into the generated task config.
    assert session.config.agents.count == 2
    assert session.config.agents.timeout_s == 300
    assert session.config.budget.rounds == 3
    assert "bias_pct within -8..8" in session.config.guardrails

    bad = session.execute("write_baseline", {"code": "def broken(:\n"})
    assert "syntax error" in bad["error"]
    good = session.execute("write_baseline", {"code": "print('model')\n"})
    assert good["ok"] is True
    written = Path(good["written_to"])
    assert written.read_text() == "print('model')\n"
    assert written == session.config.resolve(session.config.workspace.seed) / "solution/train.py"
