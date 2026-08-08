from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Idea(BaseModel):
    title: str
    hypothesis: str
    instructions: str
    category: str = "other"
    skills_used: list[str] = Field(default_factory=list)


class Attempt(BaseModel):
    id: str
    round: int
    idea: Idea
    status: Literal["running", "passed", "rejected", "failed"]
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
