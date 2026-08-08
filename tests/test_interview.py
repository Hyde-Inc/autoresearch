import asyncio
from pathlib import Path

import pandas as pd
import yaml

from autoresearch.config import load_config
from autoresearch.interview import ResearchInterview, apply_brief


def _task(tmp_path: Path) -> Path:
    seed = tmp_path / "seed/data"
    seed.mkdir(parents=True)
    pd.DataFrame(
        {
            "sku": ["a", "a", "b", "b"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
            "demand": [0, 2, 3, 4],
        }
    ).to_parquet(seed / "train.parquet", index=False)
    config = {
        "name": "test",
        "description": "Forecast demand.",
        "goal": "Lower WMAPE.",
        "director": {"model": "openrouter/test/model"},
        "agents": {"model": "openrouter/test/model"},
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_interview_returns_and_applies_brief(tmp_path: Path) -> None:
    config = load_config(_task(tmp_path))

    async def complete(_system: str, user: str) -> dict:
        assert '"zero_share": 0.25' in user
        return {
            "message": "The brief is ready.",
            "done": True,
            "brief": {
                "goal": "Reduce WMAPE by 10%.",
                "context": "Stockouts cost more than overstock.",
                "guardrails": ["runtime_s<=60"],
                "idea_hints": ["Try global tree models"],
                "metric_name": "wmape",
                "metric_description": "",
            },
        }

    interview = ResearchInterview(config, complete)
    turn = asyncio.run(
        interview.next_turn([{"role": "user", "content": "Improve our forecast."}])
    )
    assert turn.brief is not None
    apply_brief(config, turn.brief)
    updated = load_config(config.config_path)
    assert updated.goal == "Reduce WMAPE by 10%."
    assert updated.context.startswith("Stockouts")
