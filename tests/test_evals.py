from pathlib import Path

import pandas as pd
import yaml

from autoresearch.director import ResearchDirector
from autoresearch.evals import error_summary, worst_items
from autoresearch.store import RunStore


def _frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-03"])
    return pd.DataFrame(
        {
            "sku": ["a"] * 3 + ["b"] * 3,
            "date": list(dates) * 2,
            "demand": [10.0, 10.0, 10.0, 100.0, 100.0, 100.0],
            "forecast": [12.0, 12.0, 12.0, 60.0, 60.0, 60.0],
        }
    )


def test_error_summary_breaks_down_errors() -> None:
    result = error_summary(_frame(), "sku", "date", "demand")
    assert result["overall"]["wmape"] == round(126 / 330, 4)
    assert result["items_over_forecast"] == 1
    assert result["items_under_forecast"] == 1
    assert len(result["wmape_by_weekday"]) == 3


def test_worst_items_ranks_by_error_contribution() -> None:
    result = worst_items(_frame(), "sku", "date", "demand", limit=1)
    assert result["total_items"] == 2
    worst = result["worst"][0]
    assert worst["item"] == "b"
    assert worst["direction"] == "under"
    assert worst["share_of_all_error"] > 0.9


def test_director_analysis_tools_read_data_and_stored_errors(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    seed = tmp_path / "seed"
    (seed / "data").mkdir(parents=True)
    _frame().drop(columns="forecast").to_parquet(seed / "data" / "train.parquet", index=False)
    config_path = tmp_path / "task.yaml"
    config_path.write_text(yaml.safe_dump({"name": "analysis-test"}))

    from autoresearch.config import load_config

    director = ResearchDirector(load_config(config_path))
    store = RunStore(tmp_path / "run")
    store.save_validation_frame("baseline", _frame())

    profile = director._run_analysis_tool(store, "explore_training_data", {"analysis": "profile"})
    assert profile["rows"] == 6

    errors = director._run_analysis_tool(store, "analyze_errors", {"attempt_id": "baseline"})
    assert errors["overall"]["wmape"] == round(126 / 330, 4)

    missing = director._run_analysis_tool(store, "analyze_errors", {"attempt_id": "nope"})
    assert missing["available"] == ["baseline"]

    worst = director._run_analysis_tool(store, "worst_items", {"attempt_id": "baseline", "limit": 1})
    assert worst["worst"][0]["item"] == "b"
