import asyncio
from pathlib import Path

import pytest
import yaml

from autoresearch.config import load_config
from autoresearch.director import ResearchDirector, parse_json_object
from autoresearch.models import Idea
from autoresearch.skills import SelectedSkill, SkillSelection


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('{"ideas": []}', []),
        ('```json\n{"ideas": [{"title": "x"}]}\n```', [{"title": "x"}]),
        ('Here is the result: {"ideas": [1]}', [1]),
    ],
)
def test_parse_json_object(text: str, value: list) -> None:
    assert parse_json_object(text)["ideas"] == value


def test_idea_tracks_skills() -> None:
    idea = Idea(
        title="Calibrate bias",
        hypothesis="Calibration will reduce bias.",
        instructions="Apply a residual correction.",
        skills_used=["ensembling"],
    )
    assert idea.skills_used == ["ensembling"]


def test_opening_round_pins_skills_then_routes_normally(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"name": "t", "opening_round": {"skills": ["statistical-demand-models"]}}
        )
    )
    director = ResearchDirector(load_config(config_path))
    prompts: list[str] = []

    async def fake_completion(_system: str, user: str) -> dict:
        prompts.append(user)
        return {
            "ideas": [
                {
                    "title": "Seasonal naive per item",
                    "hypothesis": "h",
                    "instructions": "i",
                    "skills_used": ["statistical-demand-models", "boosting-demand-models"],
                }
            ]
        }

    monkeypatch.setattr(director, "_json_completion", fake_completion)

    def unreachable(*_args, **_kwargs):
        raise AssertionError("the opening round must not call the skill router")

    monkeypatch.setattr("autoresearch.director.select_skills", unreachable)
    ideas = asyncio.run(director.propose(count=1, round_number=1, attempts=[], notes=""))

    names = [item.name for item in director.last_skill_selection.selected]
    assert names == ["statistical-demand-models"]
    # A boosting skill the director asked for is dropped, so the agent's
    # EXPERIMENT.md carries only the pinned playbook.
    assert ideas[0].skills_used == ["statistical-demand-models"]
    assert "no gradient boosting" in prompts[0]
    assert "ExponentialSmoothing" in prompts[0]

    async def fake_select(_situation, _skills, _complete, maximum: int = 4) -> SkillSelection:
        return SkillSelection(
            selected=[SelectedSkill(name="boosting-demand-models", reason="round 2 is open")]
        )

    monkeypatch.setattr("autoresearch.director.select_skills", fake_select)
    asyncio.run(director.propose(count=1, round_number=2, attempts=[], notes=""))

    names = [item.name for item in director.last_skill_selection.selected]
    assert names == ["boosting-demand-models"]
    assert "Opening-round constraint" not in prompts[1]


def test_unknown_opening_round_skill_fails_before_the_run(tmp_path: Path) -> None:
    config_path = tmp_path / "task.yaml"
    config_path.write_text(
        yaml.safe_dump({"name": "t", "opening_round": {"skills": ["statistical-models"]}})
    )
    with pytest.raises(ValueError, match="statistical-models"):
        load_config(config_path)
