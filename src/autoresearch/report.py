from __future__ import annotations

from pathlib import Path

from .store import RunStore


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
    ]
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
