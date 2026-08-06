import numpy as np

from autoresearch.harness import forecasting_metrics


def test_forecasting_metrics() -> None:
    metrics = forecasting_metrics(
        np.array([10.0, 20.0, 30.0]), np.array([11.0, 18.0, 33.0])
    )
    assert metrics["wmape"] == 0.1
    assert metrics["rmse"] > 0
    assert metrics["bias_pct"] > 0
