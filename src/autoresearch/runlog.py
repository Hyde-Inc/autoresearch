from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import Attempt, Idea

COLORS = ("cyan", "magenta", "yellow", "blue", "green", "bright_cyan")


def _shorten(value: Any, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_target(part: dict[str, Any]) -> str:
    state = part.get("state") or {}
    tool_input = state.get("input") or {}
    if not isinstance(tool_input, dict):
        return ""
    for key in ("filePath", "path", "command", "query", "pattern", "description"):
        if tool_input.get(key):
            return _shorten(tool_input[key])
    return ""


def _display_path(value: str, workspace: Path | None) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if workspace and path.is_absolute():
        try:
            return str(path.resolve().relative_to(workspace.resolve()))
        except ValueError:
            return f"outside workspace: {path}"
    marker = "/solution/"
    if marker in value:
        return "solution/" + value.split(marker, 1)[1]
    return _shorten(value)


def format_agent_event(event: dict[str, Any], workspace: Path | None = None) -> str | None:
    """Turn an OpenCode JSON event into one useful line for a human."""
    event_type = event.get("type")
    part = event.get("part") or {}
    if event_type == "runtime_retry":
        return (
            f"OpenCode startup failed, retrying "
            f"({event.get('attempt')}/{event.get('max_attempts')})"
        )
    if event_type == "tool_use":
        tool = str(part.get("tool", "tool")).lower()
        state = part.get("state") or {}
        status = state.get("status")
        target = _tool_target(part)
        code_tools = {"edit", "write", "apply_patch"}
        check_tools = {"bash", "shell"}
        if status == "error" and tool in code_tools | check_tools:
            error = _shorten(state.get("error", "unknown error"))
            return f"{tool} failed: {error}"
        if tool in code_tools:
            path = _display_path(target, workspace)
            if path.startswith("outside workspace:"):
                return f"blocked unsafe edit {path}"
            return f"changed {path}".rstrip()
        if tool in check_tools and any(
            marker in target.lower()
            for marker in ("train.py", "pytest", "autoresearch_", "forecast", "wmape")
        ):
            return "running a local model check"
        return None
    if event_type == "text":
        raw = str(part.get("text") or "")
        lower = raw.lower()
        useful = (
            "achieves",
            "outperform",
            "improv",
            "leakage",
            "worse than",
            "validated",
            "failed",
            "error",
            "completed in",
        )
        if not any(marker in lower for marker in useful):
            return None
        text = _shorten(raw, 180)
        return f"research note: {text}" if text else None
    if event_type == "error":
        return f"agent error: {_shorten(json.dumps(event.get('error', {})), 180)}"
    return None


class ResearchRunLogger:
    def __init__(self, activity_path: Path, console: Console | None = None):
        self.console = console or Console()
        self.activity_path = activity_path
        self.activity_path.parent.mkdir(parents=True, exist_ok=True)
        self._colors: dict[str, str] = {}
        self._last_event: dict[str, str] = {}
        self._seen_events: dict[str, set[str]] = {}

    def _color(self, attempt_id: str) -> str:
        if attempt_id not in self._colors:
            self._colors[attempt_id] = COLORS[len(self._colors) % len(COLORS)]
        return self._colors[attempt_id]

    def _save(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        with self.activity_path.open("a") as handle:
            handle.write(f"{timestamp} {message}\n")

    def message(self, message: str, *, style: str = "") -> None:
        self.console.print(message, style=style, markup=False)
        self._save(message)

    def agent(self, attempt_id: str, message: str) -> None:
        prefix = f"[{attempt_id}] "
        self.console.print(Text.assemble((prefix, f"bold {self._color(attempt_id)}"), message))
        self._save(prefix + message)

    def event(
        self, attempt_id: str, event: dict[str, Any], workspace: Path | None = None
    ) -> None:
        message = format_agent_event(event, workspace)
        seen = self._seen_events.setdefault(attempt_id, set())
        if not message or self._last_event.get(attempt_id) == message or message in seen:
            return
        self._last_event[attempt_id] = message
        seen.add(message)
        self.agent(attempt_id, message)

    def run_start(
        self,
        *,
        goal: str,
        primary_metric: str,
        baseline: float,
        parallel: int,
        maximum: int,
    ) -> None:
        self.console.rule("[bold]Autoresearch lab")
        self.message(f"Goal: {goal}")
        self.message(f"Starting canvas: {primary_metric}={baseline:.6f}")
        self.message(f"Research setup: {parallel} parallel agents, up to {maximum} experiments")

    def round_plan(
        self, round_number: int, assignments: list[tuple[str, Idea]]
    ) -> None:
        self.console.rule(f"[bold cyan]Round {round_number}: proposed brush strokes")
        table = Table(show_lines=True)
        table.add_column("Agent", no_wrap=True)
        table.add_column("Strategy")
        table.add_column("Why try it")
        table.add_column("Model change")
        for attempt_id, idea in assignments:
            table.add_row(
                Text(attempt_id, style=f"bold {self._color(attempt_id)}"),
                idea.title,
                idea.hypothesis,
                idea.instructions,
            )
            self._save(
                f"[{attempt_id}] strategy={idea.title} | why={idea.hypothesis} "
                f"| change={idea.instructions}"
            )
        self.console.print(table)

    def attempt_result(
        self,
        attempt: Attempt,
        *,
        primary_metric: str,
        incumbent_value: float,
        direction: str,
    ) -> None:
        if attempt.status == "failed":
            self.agent(attempt.id, f"failed: {attempt.error or 'unknown error'}")
            return
        if attempt.status == "rejected":
            reason = attempt.error or "; ".join(attempt.guardrail_failures)
            self.agent(attempt.id, f"rejected by evaluation: {reason}")
            return
        value = attempt.metrics[primary_metric]
        denominator = max(abs(incumbent_value), 1e-12)
        improvement = (
            (incumbent_value - value) / denominator
            if direction == "min"
            else (value - incumbent_value) / denominator
        )
        outcome = "improved" if improvement > 0 else "did not improve"
        self.agent(
            attempt.id,
            f"canvas result: {primary_metric}={value:.6f}, {outcome} incumbent "
            f"by {abs(improvement) * 100:.2f}%, guardrails passed",
        )

    def round_summary(
        self, round_number: int, attempts: list[Attempt], primary_metric: str
    ) -> None:
        self.console.rule(f"[bold]Round {round_number}: evaluation summary")
        table = Table()
        table.add_column("Agent")
        table.add_column("Strategy")
        table.add_column("Status")
        table.add_column(primary_metric.upper(), justify="right")
        table.add_column("Holdout", justify="right")
        for attempt in attempts:
            metric = attempt.metrics.get(primary_metric)
            holdout = attempt.holdout_metrics.get(primary_metric)
            table.add_row(
                Text(attempt.id, style=f"bold {self._color(attempt.id)}"),
                attempt.idea.title,
                attempt.status,
                f"{metric:.6f}" if metric is not None else "-",
                f"{holdout:.6f}" if holdout is not None else "-",
            )
        self.console.print(table)
