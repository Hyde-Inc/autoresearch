import asyncio
from pathlib import Path

import yaml

from autoresearch.cli import parse_review_reply
from autoresearch.config import load_config
from autoresearch.models import Idea
from autoresearch.orchestrator import ReviewDecision, _propose_round
from autoresearch.skills import SkillSelection
from autoresearch.store import RunStore


def test_parse_review_reply_accepts_natural_phrases() -> None:
    assert parse_review_reply("Go ahead!").action == "approve"
    assert parse_review_reply("looks good").action == "approve"
    assert parse_review_reply("execute").action == "approve"
    assert parse_review_reply("Stop.").action == "stop"
    decision = parse_review_reply("approve but drop the ensemble idea")
    assert decision.action == "revise"
    assert decision.feedback == "approve but drop the ensemble idea"


def _idea(title: str) -> Idea:
    return Idea(title=title, hypothesis="h", instructions="i")


def _config(tmp_path: Path):
    task_root = tmp_path / "repo" / ".autoresearch" / "task"
    task_root.mkdir(parents=True)
    path = task_root / "task.yaml"
    path.write_text(yaml.safe_dump({"name": "t", "goal": "reduce wmape"}))
    return load_config(path)


class StubDirector:
    """Returns canned proposals and records the feedback it was given."""

    def __init__(self, batches: list[list[Idea]]):
        self.batches = batches
        self.calls: list[dict] = []
        self.last_analysis: list[str] = []
        self.last_skill_selection = SkillSelection()

    async def propose(self, **kwargs) -> list[Idea]:
        self.calls.append(kwargs)
        return self.batches[min(len(self.calls) - 1, len(self.batches) - 1)]


def test_review_approve_runs_the_plan_file(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config(tmp_path)
    director = StubDirector([[_idea("a"), _idea("b")]])
    seen: dict = {}

    def review(round_number: int, ideas: list[Idea], plan_path: Path) -> ReviewDecision:
        seen["plan_path"] = plan_path
        assert plan_path.exists()
        return ReviewDecision("approve")

    ideas, plan_path = asyncio.run(
        _propose_round(director, store, config, 2, "", 3, review=review)
    )
    assert [idea.title for idea in ideas] == ["a", "b"]
    assert director.calls[0]["count_range"] == (2, 5)
    assert director.calls[0]["count"] is None
    # The plan file lives in the repo's plans folder and is marked executed.
    assert plan_path == seen["plan_path"]
    assert plan_path.parent == tmp_path / "repo" / ".autoresearch" / "plans"
    assert "status: executed" in plan_path.read_text()


def test_review_approve_honors_hand_edits(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config(tmp_path)
    director = StubDirector([[_idea("a"), _idea("b")]])

    def review(round_number: int, ideas: list[Idea], plan_path: Path) -> ReviewDecision:
        # Human deletes experiment 2 and edits a guardrail before approving.
        text = plan_path.read_text()
        text = text[: text.index("## Experiment 2:")]
        text = text.replace("guardrails: []", 'guardrails: ["runtime_s<=120"]')
        plan_path.write_text(text)
        return ReviewDecision("approve")

    ideas, _ = asyncio.run(_propose_round(director, store, config, 2, "", 3, review=review))
    assert [idea.title for idea in ideas] == ["a"]
    assert config.guardrails == ["runtime_s<=120"]


def test_review_feedback_revises_then_approves(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config(tmp_path)
    director = StubDirector([[_idea("first")], [_idea("revised"), _idea("extra")]])
    decisions = iter(
        [ReviewDecision("revise", feedback="drop the ensemble"), ReviewDecision("approve")]
    )

    ideas, plan_path = asyncio.run(
        _propose_round(director, store, config, 3, "", 3, review=lambda r, i, p: next(decisions))
    )
    assert [idea.title for idea in ideas] == ["revised", "extra"]
    assert director.calls[1]["feedback"] == "drop the ensemble"
    assert [idea.title for idea in director.calls[1]["previous"]] == ["first"]
    # The revision rewrote the same plan file rather than creating a second one.
    assert len(list(plan_path.parent.glob("*.md"))) == 1


def test_review_stop_ends_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config(tmp_path)
    director = StubDirector([[_idea("a"), _idea("b")]])
    ideas, _ = asyncio.run(
        _propose_round(
            director, store, config, 2, "", 3, review=lambda r, i, p: ReviewDecision("stop")
        )
    )
    assert ideas == []


def test_no_reviewer_still_records_the_plan(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    config = _config(tmp_path)
    director = StubDirector([[_idea("a"), _idea("b"), _idea("c")]])
    ideas, plan_path = asyncio.run(
        _propose_round(director, store, config, 2, "", 3, review=None)
    )
    assert len(ideas) == 3
    assert director.calls[0]["count"] == 3
    assert director.calls[0]["count_range"] is None
    # Autonomous runs still archive the plan for the lab notebook.
    assert plan_path.exists()
    assert "status: executed" in plan_path.read_text()
