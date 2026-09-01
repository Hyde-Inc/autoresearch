from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """One reason an idea came to mind, tied to a checkable source.

    ``kind`` says where the reason comes from: an analysis tool run (source is
    an AnalysisRecord id like "A2"), a skill playbook (source is the skill
    name), a prior attempt (source is the attempt title), or something the
    user said (no source needed).
    """

    kind: Literal["analysis", "skill", "prior_attempt", "user_context"] = "analysis"
    source: str = ""
    observation: str


class AnalysisRecord(BaseModel):
    """Verbatim output of one analysis tool call the director made while
    designing a round, labeled with a citable id (A1, A2, ...)."""

    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        rendered = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return f"{self.tool}({rendered})"


class Idea(BaseModel):
    title: str
    hypothesis: str
    instructions: str
    category: str = "other"
    skills_used: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


def filter_evidence(idea: Idea, analysis_ids: set[str], skill_names: set[str]) -> None:
    """Drop evidence whose reference does not exist (hallucinated analysis ids
    or skill names), so everything shown to the user is checkable."""
    kept = []
    for item in idea.evidence:
        if not item.observation.strip():
            continue
        if item.kind == "analysis" and item.source and item.source not in analysis_ids:
            continue
        if item.kind == "skill" and item.source not in skill_names:
            continue
        kept.append(item)
    idea.evidence = kept


class Attempt(BaseModel):
    id: str
    round: int
    idea: Idea
    status: Literal["running", "passed", "rejected", "failed", "cancelled"]
    branch: str | None = None
    commit: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    holdout_metrics: dict[str, float] = Field(default_factory=dict)
    guardrail_failures: list[str] = Field(default_factory=list)
    error: str | None = None
    diff_path: str | None = None
    log_path: str | None = None
    promoted: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_s: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
