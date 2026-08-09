"""Research plans as editable markdown files, Cursor plan-mode style.

Every round the director writes its proposal to ``<repo>/.autoresearch/plans/``
as a markdown file with YAML frontmatter (the run parameters) and one section
per experiment. The human edits the file freely - reword hypotheses, delete or
add experiments, change frontmatter - and the edited file is what actually
runs. After the round, results are appended, so the folder is a permanent lab
notebook of every research idea ever tried and how it scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import Attempt, Idea

# Frontmatter keys the human may edit that flow back into the run.
OVERRIDE_KEYS = ("goal", "metric", "n_agents", "guardrails", "timeout_s")


class PlanError(ValueError):
    """A plan file could not be parsed; the message is safe to show the user."""


@dataclass
class ParsedPlan:
    ideas: list[Idea]
    overrides: dict = field(default_factory=dict)


def plans_dir_for_task(task_root: Path) -> Path:
    """Plans folder for a task: <repo>/.autoresearch/plans for repo-based tasks,
    <task>/plans for standalone task directories."""
    if task_root.parent.name == ".autoresearch":
        directory = task_root.parent / "plans"
    else:
        directory = task_root / "plans"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def next_plan_path(directory: Path, round_number: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    taken = [
        int(match.group(1))
        for item in directory.glob("*.md")
        if (match := re.match(r"(\d{3})-", item.name))
    ]
    number = max(taken, default=0) + 1
    return directory / f"{number:03d}-round-{round_number}.md"


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
        match = re.search(
            rf"^###\s*{name}\s*\n(.*?)(?=^###\s|\Z)", body, re.MULTILINE | re.DOTALL
        )
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


def mark_executed(path: Path) -> None:
    _set_status(path, "executed")


def append_results(path: Path, attempts: list[Attempt], metric_name: str) -> None:
    """Record how the round went inside its plan file: the lab-notebook entry."""
    if not path.exists():
        return
    _set_status(path, "completed")
    lines = ["", "## Results", ""]
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
    with path.open("a") as handle:
        handle.write("\n".join(lines))
