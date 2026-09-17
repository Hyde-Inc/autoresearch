from __future__ import annotations

import numpy as np
import pandas as pd

from autoresearch.config import DataConfig, MetricConfig, TaskConfig
from autoresearch.review import build_review, improvement


def _config(direction: str = "min") -> TaskConfig:
    return TaskConfig(
        name="demo",
        metric=MetricConfig(name="wmape", direction=direction),
        data=DataConfig(id_column="sku", date_column="date", target_column="demand"),
    )


def _frames(seed: int, base_scale: float, cand_scale: float):
    """Build baseline/candidate validation frames with covariates for cuts."""
    rng = np.random.default_rng(seed)
    ids = [f"sku{i}" for i in range(45)]
    dates = pd.date_range("2026-03-12", periods=7)
    verticals = ["grocery", "apparel", "electronics"]
    rows = []
    for i, sku in enumerate(ids):
        level = rng.integers(5, 60)
        price = float(rng.uniform(10, 400))
        vert = verticals[i % len(verticals)]
        for d in dates:
            actual = max(0.0, level + rng.normal(0, 4))
            rows.append(
                {
                    "sku": sku,
                    "date": d,
                    "demand": actual,
                    "selling_price": price,
                    "vertical": vert,
                    "oos_flag": int(rng.random() < 0.1),
                }
            )
    df = pd.DataFrame(rows)

    def with_forecast(scale: float) -> pd.DataFrame:
        out = df.copy()
        out["forecast"] = np.clip(out["demand"] + rng.normal(0, scale, len(out)), 0, None)
        return out

    return with_forecast(base_scale), with_forecast(cand_scale)


def test_improvement_respects_direction() -> None:
    assert improvement(0.5, 0.4, "min") > 0  # lower is better
    assert improvement(0.4, 0.5, "min") < 0
    assert improvement(0.4, 0.5, "max") > 0  # higher is better
    assert improvement(0.0, 0.4, "min") is None  # no baseline to divide by


def test_review_discovers_dimensions_and_approves_a_better_model() -> None:
    baseline, candidate = _frames(seed=0, base_scale=10.0, cand_scale=4.0)
    report = build_review(_config(), baseline, candidate, "a1")

    # Every step is present and populated from the discovered covariates.
    titles = {step.key for step in report.steps}
    assert {"group", "cut", "temporal"} <= titles
    block_labels = {block.label for step in report.steps for block in step.blocks}
    assert "By sku volume" in block_labels
    assert "By vertical" in block_labels
    assert "Weekday vs weekend" in block_labels
    assert "By selling_price" in block_labels
    assert "By oos_flag" in block_labels
    assert "By evaluation period" in block_labels

    # A uniformly better candidate should be recommended.
    assert report.overall_candidate < report.overall_baseline
    assert report.verdict == "Ready to merge"
    assert all(dim.status == "Pass" for dim in report.dimensions)


def test_review_flags_a_worse_model() -> None:
    baseline, candidate = _frames(seed=3, base_scale=4.0, cand_scale=12.0)
    report = build_review(_config(), baseline, candidate, "a2")
    assert report.verdict == "Not recommended"
    assert report.aggregate_score < 55


def test_review_handles_minimal_columns() -> None:
    """With only id/date/target/forecast, volume + weekday + temporal still work."""
    baseline, candidate = _frames(seed=1, base_scale=9.0, cand_scale=5.0)
    keep = ["sku", "date", "demand", "forecast"]
    report = build_review(_config(), baseline[keep], candidate[keep], "a3")
    block_labels = {block.label for step in report.steps for block in step.blocks}
    assert "By sku volume" in block_labels
    assert "Weekday vs weekend" in block_labels
    assert "By evaluation period" in block_labels
    # No covariates -> no category/price/flag cuts discovered.
    assert "By vertical" not in block_labels
    assert "By selling_price" not in block_labels
