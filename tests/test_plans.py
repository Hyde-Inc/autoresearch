from pathlib import Path

import pytest

from autoresearch.models import Attempt, Idea
from autoresearch.plans import (
    PlanError,
    findings_path,
    new_session_dir,
    parse_plan,
    plan_path,
    plans_dir_for_task,
    refresh_session_readme,
    render_plan,
    write_findings,
)


def _idea(title: str, skills: list[str] | None = None) -> Idea:
    return Idea(
        title=title,
        hypothesis=f"{title} should reduce error",
        instructions=f"Implement {title} in solution/train.py.",
        category="model",
        skills_used=skills or [],
    )


def _write(tmp_path: Path, ideas: list[Idea], **kwargs) -> Path:
    path = plan_path(tmp_path, kwargs.pop("round_number", 1))
    path.write_text(render_plan(round_number=1, ideas=ideas, **kwargs))
    return path


def test_render_parse_roundtrip(tmp_path: Path) -> None:
    ideas = [_idea("Global XGBoost", ["boosting-demand-models"]), _idea("Croston routing")]
    path = _write(
        tmp_path,
        ideas,
        goal="reduce wmape",
        metric="wmape",
        baseline="models/arima.py",
        guardrails=["runtime_s<=600"],
        timeout_s=1200,
        budget_s=3600,
        analysis=["weekday bias is worst on Mondays"],
    )
    parsed = parse_plan(path)
    assert [idea.title for idea in parsed.ideas] == ["Global XGBoost", "Croston routing"]
    assert parsed.ideas[0].skills_used == ["boosting-demand-models"]
    assert parsed.ideas[0].hypothesis == "Global XGBoost should reduce error"
    assert parsed.overrides["goal"] == "reduce wmape"
    assert parsed.overrides["n_agents"] == 2
    assert parsed.overrides["timeout_s"] == 1200
    assert parsed.overrides["budget_s"] == 3600


def test_session_folders_are_numbered_and_named_after_the_goal(tmp_path: Path) -> None:
    first = new_session_dir(tmp_path, "Reduce WMAPE!")
    second = new_session_dir(tmp_path, "Reduce WMAPE!")
    third = new_session_dir(tmp_path)
    assert first.name == "001-reduce-wmape"
    assert second.name == "002-reduce-wmape"
    assert third.name == "003-research"
    assert first.parent == tmp_path and second.is_dir() and third.is_dir()
    assert plan_path(first, 1).name == "round-1-plan.md"
    assert plan_path(first, 2).name == "round-2-plan.md"
    assert findings_path(plan_path(first, 1)).name == "round-1-findings.md"


def test_plans_dir_is_visible_research_folder(tmp_path: Path) -> None:
    repo_task = tmp_path / "repo" / ".autoresearch" / "task"
    repo_task.mkdir(parents=True)
    assert plans_dir_for_task(repo_task) == tmp_path / "repo" / "research"
    standalone = tmp_path / "tasks" / "my-task"
    standalone.mkdir(parents=True)
    assert plans_dir_for_task(standalone) == standalone / "research"


def test_session_readme_indexes_rounds_with_status_and_links(tmp_path: Path) -> None:
    session = new_session_dir(tmp_path, "reduce wmape")
    idea = _idea("Global XGBoost")
    plan = plan_path(session, 1)
    plan.write_text(render_plan(round_number=1, ideas=[idea], goal="reduce wmape", metric="wmape"))
    readme = refresh_session_readme(session)
    text = readme.read_text()
    assert "# Research session: reduce wmape" in text
    assert "[round-1-plan.md](round-1-plan.md)" in text
    assert "awaiting your review" in text
    attempt = Attempt(id="a1", round=1, idea=idea, status="passed", metrics={"wmape": 0.1})
    write_findings(plan, [attempt], "wmape", reflection="done")
    text = readme.read_text()  # write_findings refreshes the README in place
    assert "completed" in text
    assert "[round-1-findings.md](round-1-findings.md)" in text


def test_human_edits_survive(tmp_path: Path) -> None:
    path = _write(tmp_path, [_idea("Keep me"), _idea("Delete me")], goal="reduce wmape")
    text = path.read_text()
    # Human deletes the second experiment, rewords the first hypothesis, adds a
    # third experiment by hand, and changes n_agents in the frontmatter.
    text = text[: text.index("## Experiment 2:")]
    text = text.replace("Keep me should reduce error", "a reworded hypothesis")
    text = text.replace("n_agents: 2", "n_agents: 4")
    text += (
        "\n## Experiment 9: Hand-added ensemble\n\n"
        "- category: ensemble\n- skills: ensembling\n\n"
        "### Hypothesis\n\nBlending helps.\n\n"
        "### Instructions\n\nAverage the two best models.\n"
    )
    path.write_text(text)
    parsed = parse_plan(path)
    assert [idea.title for idea in parsed.ideas] == ["Keep me", "Hand-added ensemble"]
    assert parsed.ideas[0].hypothesis == "a reworded hypothesis"
    assert parsed.ideas[1].skills_used == ["ensembling"]
    assert parsed.overrides["n_agents"] == 4


def test_malformed_plans_raise_fixable_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("---\ngoal: x\n---\n\n# Plan\n\nno experiments here\n")
    with pytest.raises(PlanError, match="no experiments"):
        parse_plan(path)
    path.write_text(
        "---\ngoal: x\n---\n\n## Experiment 1: Missing parts\n\n### Hypothesis\n\nonly this\n"
    )
    with pytest.raises(PlanError, match="Missing parts"):
        parse_plan(path)
    with pytest.raises(PlanError, match="not found"):
        parse_plan(tmp_path / "absent.md")


def test_write_findings_creates_separate_file_and_completes_plan(tmp_path: Path) -> None:
    idea = _idea("Global XGBoost")
    path = _write(tmp_path, [idea], goal="reduce wmape", metric="wmape")
    attempt = Attempt(
        id="a1",
        round=1,
        idea=idea,
        status="passed",
        metrics={"wmape": 0.101},
        holdout_metrics={"wmape": 0.117},
        promoted=True,
    )
    findings = write_findings(path, [attempt], "wmape", reflection="XGBoost won; scale it.")
    assert findings == findings_path(path)
    text = findings.read_text()
    assert text.startswith("# Round 1 findings")
    assert "0.101000" in text
    assert "| yes |" in text
    assert "XGBoost won; scale it." in text
    # The plan file itself is marked completed and still parses.
    assert "status: completed" in path.read_text()
    parsed = parse_plan(path)
    assert [item.title for item in parsed.ideas] == ["Global XGBoost"]
    # A missing plan file is a no-op, not a crash.
    assert write_findings(tmp_path / "absent.md", [attempt], "wmape") is None
