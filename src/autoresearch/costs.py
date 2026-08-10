"""Token-cost accounting from OpenCode agent logs.

OpenCode reports the paid cost of each model turn in its streamed JSON events:
a ``step_finish`` event carries ``part.cost`` (USD for that step) and
``part.tokens`` (input / output / reasoning counts). Nothing else in the
pipeline spends model tokens - training and evaluation run locally - so summing
these events across an attempt's log yields that experiment's full spend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CostSummary:
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: CostSummary) -> CostSummary:
        return CostSummary(
            self.cost_usd + other.cost_usd,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


def event_cost(event: object) -> float:
    """USD cost carried by one ``step_finish`` event, or 0.0 for anything else."""
    if not isinstance(event, dict) or event.get("type") != "step_finish":
        return 0.0
    part = event.get("part") if isinstance(event.get("part"), dict) else {}
    try:
        return float(part.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _tokens(part: dict) -> tuple[int, int, int]:
    tokens = part.get("tokens") if isinstance(part.get("tokens"), dict) else {}

    def count(key: str) -> int:
        try:
            return int(tokens.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return count("input"), count("output"), count("reasoning")


def log_cost(log_path: Path | None) -> CostSummary:
    """Total cost and tokens recorded in one agent's JSONL event log."""
    summary = CostSummary()
    if log_path is None or not Path(log_path).exists():
        return summary
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{") or '"step_finish"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") != "step_finish":
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        try:
            summary.cost_usd += float(part.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass
        tin, tout, treason = _tokens(part)
        summary.input_tokens += tin
        summary.output_tokens += tout
        summary.reasoning_tokens += treason
    return summary


def format_cost(usd: float) -> str:
    """Compact USD rendering: cents get more precision than dollars."""
    if usd <= 0:
        return "$0.00"
    if usd < 1:
        return f"${usd:.4f}"
    return f"${usd:.2f}"
