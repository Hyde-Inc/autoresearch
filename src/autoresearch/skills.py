from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .config import TaskConfig

BUILTIN_SKILLS_DIR = Path(__file__).with_name("skills_library")


class ResearchSkill(BaseModel):
    name: str
    description: str
    body: str
    source: Path


class SelectedSkill(BaseModel):
    name: str
    reason: str


class SkillSelection(BaseModel):
    selected: list[SelectedSkill] = Field(default_factory=list)


def load_skill(path: Path) -> ResearchSkill:
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"skill must start with YAML frontmatter: {path}")
    _, frontmatter, body = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter) or {}
    return ResearchSkill(
        name=metadata["name"],
        description=metadata["description"],
        body=body.strip(),
        source=path,
    )


def _skill_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.md")))
        elif path.suffix == ".md" and path.exists():
            files.append(path)
    return files


def load_skills(config: TaskConfig | None = None) -> list[ResearchSkill]:
    paths = [BUILTIN_SKILLS_DIR]
    if config:
        paths.extend(config.resolve(path) for path in config.skills)
    by_name: dict[str, ResearchSkill] = {}
    for path in _skill_files(paths):
        skill = load_skill(path)
        by_name[skill.name] = skill
    return list(by_name.values())


def skill_index(skills: list[ResearchSkill]) -> str:
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)


def named_skill_context(names: Iterable[str], skills: list[ResearchSkill]) -> str:
    """Render full playbooks for known names, preserving the requested order."""
    by_name = {skill.name: skill for skill in skills}
    return "\n\n".join(
        f"## Skill: {by_name[name].name}\n\n{by_name[name].body}"
        for name in names
        if name in by_name
    )


async def select_skills(
    situation: str,
    skills: list[ResearchSkill],
    complete: Callable[[str, str], Awaitable[dict]],
    maximum: int = 4,
) -> SkillSelection:
    if not skills:
        return SkillSelection()
    system = (
        "You route an ML forecasting problem to a small library of research skills. "
        f"Choose up to {maximum} skills that directly apply. Prefer a focused selection over "
        "generic coverage. Return JSON with key 'selected', an array of objects with exactly "
        "'name' and 'reason'. Only use names from the supplied index."
    )
    payload = await complete(
        system,
        f"Situation:\n{situation}\n\nAvailable skills:\n{skill_index(skills)}",
    )
    known = {skill.name for skill in skills}
    selected = SkillSelection.model_validate(payload)
    selected.selected = [item for item in selected.selected if item.name in known][:maximum]
    return selected


def selected_skill_context(
    selection: SkillSelection, skills: list[ResearchSkill]
) -> str:
    by_name = {skill.name: skill for skill in skills}
    blocks = []
    for item in selection.selected:
        skill = by_name[item.name]
        blocks.append(
            f"## Skill: {skill.name}\nWhy selected: {item.reason}\n\n{skill.body}"
        )
    return "\n\n".join(blocks)


def selection_json(selection: SkillSelection) -> str:
    return json.dumps(selection.model_dump(mode="json"))
