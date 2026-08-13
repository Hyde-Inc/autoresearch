import asyncio
from pathlib import Path

from autoresearch.skills import (
    load_skill,
    load_skills,
    pinned_selection,
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
                {"name": "ensembling", "reason": "Bias is high."},
                {"name": "not-real", "reason": "Should be removed."},
            ]
        }

    selection = asyncio.run(select_skills("high under-forecast bias", skills, complete))
    assert [item.name for item in selection.selected] == ["ensembling"]
    assert "out-of-sample" in selected_skill_context(selection, skills)


def test_pinned_selection_keeps_order_and_drops_unknown_names() -> None:
    skills = load_skills()
    selection = pinned_selection(["statistical-demand-models", "not-real"], skills)
    assert [item.name for item in selection.selected] == ["statistical-demand-models"]
    assert "pinned" in selection.selected[0].reason
    assert "ExponentialSmoothing" in selected_skill_context(selection, skills)


def test_builtin_library_has_grounded_model_family_playbooks() -> None:
    skills = load_skills()
    by_name = {skill.name: skill for skill in skills}
    assert set(by_name) == {
        "boosting-demand-models",
        "cannibalization-effects",
        "chronos-playbook",
        "ensembling",
        "feature-selection-tree-models",
        "intermittent-demand",
        "model-ladder",
        "payday-sale-effects",
        "retail-demand-data",
        "seasonality-transition-effects",
        "statistical-demand-models",
        "temporal-validation-and-leakage",
    }

    # These are implementation playbooks, not generic model-selection blurbs.
    required_terms = {
        "boosting-demand-models": ("XGBRegressor", "lightgbm", "CatBoostRegressor"),
        "chronos-playbook": ("BaseChronosPipeline", "predict_quantiles", "Chronos2Pipeline"),
        "statistical-demand-models": ("ExponentialSmoothing", "ARIMA", "SARIMAX"),
        "intermittent-demand": ("Croston", "SBA", "TSB"),
        "retail-demand-data": ("stockout", "planned", "cold start"),
        "temporal-validation-and-leakage": ("forecast origin", "rolling origins", "leakage"),
        "ensembling": ("convex", "residual correlation", "calibration"),
    }
    for name, terms in required_terms.items():
        body = by_name[name].body
        assert len(body) >= 2_000, f"{name} is too shallow to be an implementation skill"
        for term in terms:
            assert term.lower() in body.lower(), f"{name} is missing {term}"
