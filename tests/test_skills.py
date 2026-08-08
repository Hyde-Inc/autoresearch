import asyncio
from pathlib import Path

from autoresearch.skills import (
    load_skill,
    load_skills,
    select_skills,
    selected_skill_context,
)


def test_load_skill_frontmatter(tmp_path: Path) -> None:
    path = tmp_path / "custom.md"
    path.write_text(
        "---\nname: custom\ndescription: Use for custom cases.\n---\nDo the useful thing.\n"
    )
    skill = load_skill(path)
    assert skill.name == "custom"
    assert skill.body == "Do the useful thing."


def test_select_skills_filters_unknown_names() -> None:
    skills = load_skills()

    async def complete(_system: str, _user: str) -> dict:
        return {
            "selected": [
                {"name": "bias-correction", "reason": "Bias is high."},
                {"name": "not-real", "reason": "Should be removed."},
            ]
        }

    selection = asyncio.run(select_skills("high under-forecast bias", skills, complete))
    assert [item.name for item in selection.selected] == ["bias-correction"]
    assert "out-of-sample residuals" in selected_skill_context(selection, skills)
