from __future__ import annotations

from pathlib import Path
from typing import Any

from .costs import format_cost
from .store import RunStore


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """A compact, machine-readable snapshot of a run's outcome.

    Shared by ``autoresearch report`` and the end-of-run summary so the terminal
    panel and the ``--json`` payload never drift apart.
    """
    store = RunStore(run_dir)
    state = store.load_state()
    attempts = store.load_attempts()
    metric = state.get("primary_metric", "wmape")
    direction = state.get("metric_direction", "min")
    baseline = (state.get("baseline") or {}).get(metric)
    final = (state.get("incumbent") or {}).get(metric)
    improvement = None
    if baseline not in (None, 0) and final is not None:
        improvement = (baseline - final) / baseline * 100
        if direction == "max":
            improvement = -improvement
    total_cost = sum(item.metadata.get("cost_usd", 0.0) or 0.0 for item in attempts)
    promoted = [item for item in attempts if item.promoted]
    ranked = store.leaderboard(metric, direction)
    best = ranked[0] if ranked else None
    return {
        "run_dir": str(store.run_dir),
        "task": state.get("task", store.run_dir.name),
        "goal": state.get("goal"),
        "status": state.get("status"),
        "rounds": state.get("round"),
        "metric": metric,
        "metric_direction": direction,
        "baseline": baseline,
        "final": final,
        "improvement_pct": improvement,
        "total_cost_usd": total_cost,
        "counts": {
            "completed": len(attempts),
            "passed": sum(item.status == "passed" for item in attempts),
            "rejected": sum(item.status == "rejected" for item in attempts),
            "failed": sum(item.status == "failed" for item in attempts),
            "cancelled": sum(item.status == "cancelled" for item in attempts),
            "promoted": len(promoted),
        },
        "best_attempt": None
        if best is None
        else {
            "id": best.id,
            "title": best.idea.title,
            "metric": best.metrics.get(metric),
            "promoted": best.promoted,
        },
        "promoted": [{"id": item.id, "title": item.idea.title} for item in promoted],
    }


def build_report(run_dir: Path) -> str:
    store = RunStore(run_dir)
    state = store.load_state()
    attempts = store.load_attempts()
    baseline = state.get("baseline", {})
    incumbent = state.get("incumbent", {})
    metric = state.get("primary_metric", "wmape")
    start = baseline.get(metric)
    final = incumbent.get(metric)
    improvement = ((start - final) / start * 100) if start and final is not None else None
    lines = [
        f"# Autoresearch report: {state.get('task', run_dir.name)}",
        "",
        f"Goal: {state.get('goal', 'unknown')}",
        "",
        "## Outcome",
        "",
        f"- Experiments completed: {len(attempts)}",
        f"- Passing: {sum(item.status == 'passed' for item in attempts)}",
        f"- Rejected: {sum(item.status == 'rejected' for item in attempts)}",
        f"- Failed: {sum(item.status == 'failed' for item in attempts)}",
        f"- Cancelled: {sum(item.status == 'cancelled' for item in attempts)}",
    ]
    total_cost = sum(item.metadata.get("cost_usd", 0.0) or 0.0 for item in attempts)
    if total_cost > 0 or state.get("cost_usd"):
        lines.append(f"- Total model spend: {format_cost(total_cost)}")
    if improvement is not None:
        lines.extend(
            [
                f"- Baseline {metric}: {start:.6f}",
                f"- Final {metric}: {final:.6f}",
                f"- Relative improvement: {improvement:.2f}%",
            ]
        )
    lines.extend(["", "## Promoted experiments", ""])
    promoted = [item for item in attempts if item.promoted]
    if not promoted:
        lines.append("No experiment beat the incumbent while passing every guardrail.")
    for item in promoted:
        lines.extend(
            [
                f"### {item.idea.title} (`{item.id}`)",
                "",
                item.idea.hypothesis,
                "",
                f"- Research skills: {', '.join(item.idea.skills_used) or 'none'}",
                f"- Validation metrics: {item.metrics}",
                f"- Hidden holdout metrics: {item.holdout_metrics}",
                "",
            ]
        )
    lines.extend(["## Full experiment ledger", ""])
    for item in attempts:
        result = item.metrics.get(metric)
        suffix = f", {metric}={result:.6f}" if result is not None else ""
        cost = item.metadata.get("cost_usd", 0.0) or 0.0
        if cost > 0:
            suffix += f", cost {format_cost(cost)}"
        lines.append(f"- `{item.id}` {item.idea.title}: **{item.status}**{suffix}")
        if item.idea.skills_used:
            lines.append(f"  - Skills: {', '.join(item.idea.skills_used)}")
        if item.error:
            lines.append(f"  - Error: {item.error}")
        for failure in item.guardrail_failures:
            lines.append(f"  - Guardrail: {failure}")
    if store.notes_file.exists():
        lines.extend(["", "## Research Director lab notes", "", store.notes_file.read_text().strip()])
    return "\n".join(lines).strip() + "\n"
