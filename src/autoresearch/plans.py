"""Research plans as editable markdown files, Cursor plan-mode style.

Every research session gets its own numbered, goal-named folder under the
visible ``<repo>/research/`` directory, e.g. ``research/001-reduce-wmape/``.
Inside it, each round produces two markdown files: ``round-N-plan.md``, the
plan the director proposes (YAML frontmatter with the run parameters plus one
section per experiment) which the human edits freely before typing
``execute``; and ``round-N-findings.md``, written after the round with
per-experiment results and the director's reflection. A ``README.md`` index at
the top of each session folder links every round and its status, so the folder
reads like a lab notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from .models import Attempt, Idea

# Frontmatter keys the human may edit that flow back into the run.
OVERRIDE_KEYS = ("goal", "metric", "n_agents", "guardrails", "timeout_s")

_STANDARD_STATUSES = {
    "proposed": "awaiting your review",
    "executed": "running",
    "completed": "completed",
}


class PlanError(ValueError):
    """A plan file could not be parsed; the message is safe to show the user."""


@dataclass
class ParsedPlan:
    ideas: list[Idea]
    overrides: dict = field(default_factory=dict)


def plans_dir_for_task(task_root: Path) -> Path:
    """Notebook folder for a task: the visible <repo>/research for repo-based
    tasks, <task>/research for standalone task directories."""
    if task_root.parent.name == ".autoresearch":
        directory = task_root.parent.parent / "research"
    else:
        directory = task_root / "research"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "research"


def new_session_dir(plans_root: Path, goal: str = "") -> Path:
    """Create a numbered, goal-named folder for one research session,
    e.g. research/001-reduce-wmape/."""
    plans_root.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(match.group(1))
        for entry in plans_root.iterdir()
        if entry.is_dir() and (match := re.match(r"^(\d+)-", entry.name))
    ]
    base = f"{max(numbers, default=0) + 1:03d}-{slugify(goal)}"
    directory = plans_root / base
    suffix = 2
    while directory.exists():
        directory = plans_root / f"{base}-{suffix}"
        suffix += 1
    directory.mkdir(parents=True)
    return directory


def plan_path(session_dir: Path, round_number: int) -> Path:
    return session_dir / f"round-{round_number}-plan.md"


def findings_path(plan_file: Path) -> Path:
    return plan_file.with_name(f"{plan_file.stem.removesuffix('-plan')}-findings.md")


def _round_number(plan_file: Path) -> int:
    match = re.search(r"round-(\d+)", plan_file.stem)
    return int(match.group(1)) if match else 0


def refresh_session_readme(session_dir: Path) -> Path:
    """Regenerate the session's README.md index from the round files on disk."""
    plan_files = sorted(session_dir.glob("round-*-plan.md"), key=_round_number)
    goal = metric = baseline = ""
    rows = []
    for plan_file in plan_files:
        try:
            frontmatter, _ = _split_frontmatter(plan_file.read_text())
        except PlanError:
            frontmatter = {}
        goal = str(frontmatter.get("goal") or goal)
        metric = str(frontmatter.get("metric") or metric)
        baseline = str(frontmatter.get("baseline") or baseline)
        status = str(frontmatter.get("status") or "proposed")
        findings = findings_path(plan_file)
        rows.append(
            (
                _round_number(plan_file),
                plan_file.name,
                _STANDARD_STATUSES.get(status, status),
                findings.name if findings.exists() else None,
            )
        )
    title = goal or session_dir.name.partition("-")[2].replace("-", " ") or "research session"
    lines = [
        f"# Research session: {title}",
        "",
        f"- started: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}"
        if not rows
        else f"- rounds so far: {len(rows)}",
    ]
    if metric:
        lines.append(f"- metric: {metric}")
    if baseline:
        lines.append(f"- baseline: {baseline}")
    lines += [
        "",
        "Each round has a plan file (edit it, then type `execute` in the chat) and a",
        "findings file with results and the director's reflection once the round ran.",
        "",
        "| round | plan | status | findings |",
        "|---|---|---|---|",
    ]
    for number, plan_name, status, findings_name in rows:
        findings_cell = f"[{findings_name}]({findings_name})" if findings_name else "-"
        lines.append(f"| {number} | [{plan_name}]({plan_name}) | {status} | {findings_cell} |")
    if not rows:
        lines.append("| - | no plan proposed yet | - | - |")
    lines.append("")
    destination = session_dir / "README.md"
    destination.write_text("\n".join(lines))
    return destination


def render_plan(
    *,
    round_number: int,
    ideas: list[Idea],
    goal: str = "",
    metric: str = "",
    baseline: str = "",
    n_agents: int | None = None,
    guardrails: list[str] | None = None,
    timeout_s: int | None = None,
    analysis: list[str] | None = None,
) -> str:
    frontmatter = {
        "round": round_number,
        "status": "proposed",
        "goal": goal,
        "metric": metric,
        "baseline": baseline,
        "n_agents": n_agents if n_agents is not None else len(ideas),
        "guardrails": list(guardrails or []),
        "timeout_s": timeout_s,
    }
    lines = [
        "---",
        yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip(),
        "---",
        "",
        f"# Research Plan - Round {round_number}",
        "",
        "Edit this file freely: reword hypotheses, delete or add experiments, change",
        "the frontmatter. The edited file is exactly what runs when you type `execute`.",
        "",
    ]
    if analysis:
        lines += ["## Director's analysis", ""]
        lines += [f"- {item}" for item in analysis]
        lines.append("")
    for index, idea in enumerate(ideas, 1):
        lines += [
            f"## Experiment {index}: {idea.title}",
            "",
            f"- category: {idea.category}",
            f"- skills: {', '.join(idea.skills_used) or 'none'}",
            "",
            "### Hypothesis",
            "",
            idea.hypothesis.strip(),
            "",
            "### Instructions",
            "",
            idea.instructions.strip(),
            "",
        ]
    return "\n".join(lines)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise PlanError("frontmatter is not closed; the file must start with --- ... ---")
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise PlanError(f"frontmatter is not valid YAML: {exc}") from None
    if not isinstance(frontmatter, dict):
        raise PlanError("frontmatter must be a YAML mapping of key: value lines")
    return frontmatter, parts[2]


def _parse_experiment(heading: str, body: str) -> Idea:
    title = re.sub(r"^Experiment\s*\d*\s*:?\s*", "", heading).strip() or heading.strip()
    category = "other"
    skills: list[str] = []
    for match in re.finditer(r"^-\s*(category|skills)\s*:\s*(.*)$", body, re.MULTILINE):
        key, value = match.group(1), match.group(2).strip()
        if key == "category" and value:
            category = value
        elif key == "skills" and value.lower() != "none":
            skills = [item.strip() for item in value.split(",") if item.strip()]

    def section(name: str) -> str:
        match = re.search(rf"^###\s*{name}\s*\n(.*?)(?=^###\s|\Z)", body, re.MULTILINE | re.DOTALL)
        return match.group(1).strip() if match else ""

    hypothesis = section("Hypothesis")
    instructions = section("Instructions")
    if not hypothesis or not instructions:
        raise PlanError(
            f"experiment '{title}' needs both a '### Hypothesis' and a '### Instructions' "
            "section with text under each"
        )
    return Idea(
        title=title,
        hypothesis=hypothesis,
        instructions=instructions,
        category=category,
        skills_used=skills,
    )


def parse_plan(path: Path) -> ParsedPlan:
    """Parse a plan file, tolerating human edits. Raises PlanError with a fixable message."""
    if not path.exists():
        raise PlanError(f"plan file not found: {path}")
    frontmatter, body = _split_frontmatter(path.read_text())
    overrides = {
        key: frontmatter[key]
        for key in OVERRIDE_KEYS
        if key in frontmatter and frontmatter[key] not in (None, "", [])
    }
    ideas: list[Idea] = []
    sections = re.split(r"^##\s+", body, flags=re.MULTILINE)
    for chunk in sections[1:]:
        heading, _, section_body = chunk.partition("\n")
        if re.match(r"(?i)results\b|director", heading.strip()):
            continue
        if not re.match(r"(?i)experiment\b", heading.strip()):
            continue
        ideas.append(_parse_experiment(heading, section_body))
    if not ideas:
        raise PlanError(
            "no experiments found; each one needs a '## Experiment N: title' heading with "
            "'### Hypothesis' and '### Instructions' sections"
        )
    return ParsedPlan(ideas=ideas, overrides=overrides)


def _set_status(path: Path, status: str) -> None:
    frontmatter, body = _split_frontmatter(path.read_text())
    frontmatter["status"] = status
    path.write_text(
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        + "\n---"
        + body
    )
    refresh_session_readme(path.parent)


def mark_executed(path: Path) -> None:
    _set_status(path, "executed")


def write_findings(
    plan_file: Path,
    attempts: list[Attempt],
    metric_name: str,
    reflection: str = "",
) -> Path | None:
    """Write the round's findings next to its plan file: the lab-notebook entry."""
    if not plan_file.exists():
        return None
    _set_status(plan_file, "completed")
    round_label = plan_file.stem.removesuffix("-plan").replace("-", " ").capitalize()
    lines = [f"# {round_label} findings", ""]
    header = f"| experiment | status | {metric_name} | holdout | promoted | note |"
    lines += [header, "|" + "---|" * 6]
    for attempt in attempts:
        metric = attempt.metrics.get(metric_name)
        holdout = attempt.holdout_metrics.get(metric_name)
        note = attempt.error or "; ".join(attempt.guardrail_failures) or "-"
        lines.append(
            f"| {attempt.idea.title} | {attempt.status} "
            f"| {f'{metric:.6f}' if metric is not None else '-'} "
            f"| {f'{holdout:.6f}' if holdout is not None else '-'} "
            f"| {'yes' if attempt.promoted else 'no'} "
            f"| {note[:160]} |"
        )
    lines.append("")
    if reflection.strip():
        lines += ["## Director's reflection", "", reflection.strip(), ""]
    destination = findings_path(plan_file)
    destination.write_text("\n".join(lines))
    refresh_session_readme(plan_file.parent)
    return destination
