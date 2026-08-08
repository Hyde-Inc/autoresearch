import asyncio
from pathlib import Path

from autoresearch.cli import parse_review_reply
from autoresearch.models import Idea
from autoresearch.orchestrator import ReviewDecision, _propose_round
from autoresearch.skills import SkillSelection
from autoresearch.store import RunStore


def test_parse_review_reply_accepts_natural_phrases() -> None:
    assert parse_review_reply("Go ahead!").action == "approve"
    assert parse_review_reply("looks good").action == "approve"
    assert parse_review_reply("approve").action == "approve"
    assert parse_review_reply("Stop.").action == "stop"
    decision = parse_review_reply("approve but drop the ensemble idea")
    assert decision.action == "revise"
    assert decision.feedback == "approve but drop the ensemble idea"


def _idea(title: str) -> Idea:
    return Idea(title=title, hypothesis="h", instructions="i")


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


def test_review_approve_returns_ideas(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    director = StubDirector([[_idea("a"), _idea("b")]])
    ideas = asyncio.run(
        _propose_round(
            director, store, 2, "", 3, review=lambda round_number, ideas: ReviewDecision("approve")
        )
    )
    assert [idea.title for idea in ideas] == ["a", "b"]
    assert director.calls[0]["count_range"] == (2, 5)
    assert director.calls[0]["count"] is None


def test_review_feedback_revises_then_approves(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    director = StubDirector([[_idea("first")], [_idea("revised"), _idea("extra")]])
    decisions = iter(
        [ReviewDecision("revise", feedback="drop the ensemble"), ReviewDecision("approve")]
    )

    ideas = asyncio.run(
        _propose_round(director, store, 3, "", 3, review=lambda r, i: next(decisions))
    )
    assert [idea.title for idea in ideas] == ["revised", "extra"]
    assert director.calls[1]["feedback"] == "drop the ensemble"
    assert [idea.title for idea in director.calls[1]["previous"]] == ["first"]


def test_review_stop_ends_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    director = StubDirector([[_idea("a"), _idea("b")]])
    ideas = asyncio.run(
        _propose_round(
            director, store, 2, "", 3, review=lambda round_number, ideas: ReviewDecision("stop")
        )
    )
    assert ideas == []


def test_no_reviewer_keeps_autonomous_exact_count(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    director = StubDirector([[_idea("a"), _idea("b"), _idea("c")]])
    ideas = asyncio.run(_propose_round(director, store, 2, "", 3, review=None))
    assert len(ideas) == 3
    assert director.calls[0]["count"] == 3
    assert director.calls[0]["count_range"] is None
