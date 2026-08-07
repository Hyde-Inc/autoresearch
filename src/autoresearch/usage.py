"""Parse token and cost usage from an OpenCode JSONL session log."""

from __future__ import annotations

import json
from pathlib import Path


def log_usage(log_path: Path) -> tuple[int, float]:
    """Return (total_tokens, total_cost_usd) for one agent session log."""
    if not log_path.exists():
        return 0, 0.0
    total_tokens = 0
    total_cost = 0.0
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "step_finish":
            continue
        part = event.get("part") or {}
        tokens = part.get("tokens") or {}
        total_tokens += int(tokens.get("output", 0)) + int(tokens.get("input", 0))
        cost = part.get("cost")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    return total_tokens, round(total_cost, 6)
