import pytest

from autoresearch.director import parse_json_object
from autoresearch.models import Idea


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
