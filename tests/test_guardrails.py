from autoresearch.guardrails import evaluate_guardrail


def test_baseline_multiplier() -> None:
    result = evaluate_guardrail("rmse<=baseline*1.10", {"rmse": 10.5}, {"rmse": 10.0})
    assert result.passed


def test_within_range() -> None:
    assert evaluate_guardrail(
        "bias_pct within -8..8", {"bias_pct": -2.5}, {}
    ).passed
    assert not evaluate_guardrail(
        "bias_pct within -8..8", {"bias_pct": 9.0}, {}
    ).passed


def test_invalid_and_missing_metrics_fail_closed() -> None:
    assert not evaluate_guardrail("nonsense", {}, {}).passed
    assert not evaluate_guardrail("rmse<=10", {}, {}).passed
