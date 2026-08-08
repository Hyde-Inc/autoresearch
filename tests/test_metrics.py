import pandas as pd
import pytest

from autoresearch.metrics import MetricSpec, compile_metric, validate_spec


def test_custom_metric_matches_hand_worked_example() -> None:
    spec = MetricSpec.model_validate(
        {
            "name": "mae",
            "direction": "min",
            "description": "Mean absolute error",
            "understanding": "Average absolute forecast error.",
            "example": {
                "rows": [
                    {"actual": 10, "forecast": 8},
                    {"actual": 20, "forecast": 22},
                ],
                "steps": ["Errors are 2 and 2", "Mean is 2"],
                "value": 2,
            },
            "code": (
                "def metric(df: pd.DataFrame) -> float:\n"
                "    return float((df['actual'] - df['forecast']).abs().mean())\n"
            ),
        }
    )
    validation = validate_spec(spec, ["actual", "forecast"])
    assert validation.passed
    assert compile_metric(spec.code)(
        pd.DataFrame(spec.example.rows)
    ) == pytest.approx(2)


def test_custom_metric_rejects_unsafe_import() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        compile_metric("import os\ndef metric(df):\n    return 0.0\n")
