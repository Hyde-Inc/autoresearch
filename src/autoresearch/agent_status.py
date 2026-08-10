"""Per-agent live state for the parallel-round dashboard.

Each experiment in a round gets one :class:`AgentStatus` row. While the
OpenCode coding agent works it streams JSON events; :func:`classify_event`
turns each raw event into a factual phase (``Exploring`` / ``Editing`` /
``Testing``) plus a short human-readable action such as
``read solution/train.py``. The orchestrator moves rows through the outer
lifecycle (``Setting up`` -> agent phases -> ``Committing`` ->
``Protected eval`` -> ``Passed`` / ``Failed`` / ``Rejected`` /
``Cancelled``) and the dashboard renders them.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .costs import event_cost

SETTING_UP = "Setting up"
EXPLORING = "Exploring"
EDITING = "Editing"
TESTING = "Testing"
COMMITTING = "Committing"
TRAINING = "Training"
"""The orchestrator is running solution/train.py; no coding agent is billed."""
EVALUATING = "Protected eval"
PASSED = "Passed"
FAILED = "Failed"
REJECTED = "Rejected"
CANCELLED = "Cancelled"

DONE_PHASES = frozenset({PASSED, FAILED, REJECTED, CANCELLED})

MAX_ACTION_CHARS = 64

_EXPLORE_TOOLS = frozenset({"read", "grep", "glob", "list", "ls", "webfetch", "todoread"})
_EDIT_TOOLS = frozenset({"write", "edit", "multiedit", "patch", "apply_patch"})
_TEST_HINTS = ("pytest", "python", "uv run", "test", "train", "backtest")


def _clip(value: object, limit: int = MAX_ACTION_CHARS) -> str:
    """Collapse whitespace and clip so one event can never distort the table."""
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _short_path(value: object) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    return "/".join(parts[-2:]) if parts else ""


def classify_event(event: object) -> tuple[str | None, str | None]:
    """Map one OpenCode JSON event to ``(phase, action)``.

    Either element may be ``None``: ``phase=None`` keeps the current phase and
    ``action=None`` means the event carries nothing worth showing. Unknown or
    malformed events are silently ignored.
    """
    if not isinstance(event, dict):
        return None, None
    kind = event.get("type")
    part = event.get("part") if isinstance(event.get("part"), dict) else {}

    if kind == "tool_use":
        tool = str(part.get("tool") or "").lower()
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        tool_input = state.get("input") if isinstance(state.get("input"), dict) else {}
        title = str(state.get("title") or "").strip()
        target = _short_path(
            tool_input.get("filePath")
            or tool_input.get("file_path")
            or tool_input.get("path")
            or tool_input.get("pattern")
            or ""
        )
        if tool in _EDIT_TOOLS:
            return EDITING, _clip(f"{tool} {target}" if target else title or tool)
        if tool in _EXPLORE_TOOLS:
            return EXPLORING, _clip(f"{tool} {target}" if target else title or tool)
        if tool == "bash":
            command = str(tool_input.get("command") or "")
            action = _clip(title or command or "shell command")
            if any(hint in command.lower() for hint in _TEST_HINTS):
                return TESTING, action
            return None, action
        if tool == "task":
            return EXPLORING, _clip(title or "delegating to a subagent")
        if tool:
            return None, _clip(title or tool)
        return None, None

    if kind in ("text", "reasoning"):
        snippet = str(part.get("text") or "").strip()
        if not snippet:
            return None, None
        return None, _clip(f"thinking: {snippet}")

    if kind == "error":
        error = event.get("error")
        message = ""
        if isinstance(error, dict):
            data = error.get("data") if isinstance(error.get("data"), dict) else {}
            message = str(data.get("message") or error.get("name") or "")
        return None, _clip(f"error: {message or 'unknown'}")

    # step_start / step_finish are model-turn boundaries with no user-facing info.
    return None, None


@dataclass
class AgentStatus:
    """Live, dashboard-facing state of one experiment agent."""

    index: int
    attempt_id: str
    title: str
    phase: str = SETTING_UP
    action: str = "creating worktree"
    log_path: Path | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    cancel_requested: bool = False
    cost_usd: float = 0.0
    trail: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    @property
    def done(self) -> bool:
        return self.phase in DONE_PHASES

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def elapsed(self) -> str:
        total = int(self.elapsed_s)
        return f"{total // 60:02d}:{total % 60:02d}"

    def set(self, phase: str | None = None, action: str | None = None) -> None:
        """Update live state; terminal phases are final and never overwritten."""
        if self.done:
            return
        if phase:
            self.phase = phase
        if action:
            action = _clip(action)
            self.action = action
            if not self.trail or self.trail[-1] != action:
                self.trail.append(action)

    def finish(self, phase: str, action: str = "") -> None:
        """Move to a terminal phase and freeze the elapsed timer."""
        if self.done:
            return
        self.set(phase, action or phase.lower())
        self.finished_at = time.monotonic()

    def apply_event(self, event: object) -> None:
        """Fold one streamed OpenCode event into this row."""
        self.cost_usd += event_cost(event)
        phase, action = classify_event(event)
        if phase or action:
            self.set(phase, action)
