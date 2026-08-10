import json
from pathlib import Path

from autoresearch.costs import CostSummary, event_cost, format_cost, log_cost


def _step(cost: float, tokens: dict | None = None) -> dict:
    part = {"cost": cost}
    if tokens is not None:
        part["tokens"] = tokens
    return {"type": "step_finish", "part": part}


def test_event_cost_only_counts_step_finish() -> None:
    assert event_cost(_step(0.02)) == 0.02
    assert event_cost({"type": "tool", "part": {"cost": 0.5}}) == 0.0
    assert event_cost("not-an-event") == 0.0
    assert event_cost({"type": "step_finish", "part": {"cost": None}}) == 0.0


def test_log_cost_sums_cost_and_tokens(tmp_path: Path) -> None:
    log = tmp_path / "attempt.jsonl"
    lines = [
        json.dumps(_step(0.01, {"input": 100, "output": 20, "reasoning": 5})),
        "some plain non-json diagnostic line",
        json.dumps({"type": "text", "part": {"text": "hi"}}),
        json.dumps(_step(0.02, {"input": 200, "output": 40, "reasoning": 0})),
    ]
    log.write_text("\n".join(lines) + "\n")

    summary = log_cost(log)

    assert round(summary.cost_usd, 6) == 0.03
    assert summary.input_tokens == 300
    assert summary.output_tokens == 60
    assert summary.reasoning_tokens == 5


def test_log_cost_missing_file_is_zero(tmp_path: Path) -> None:
    assert log_cost(tmp_path / "nope.jsonl") == CostSummary()
    assert log_cost(None) == CostSummary()


def test_format_cost_precision() -> None:
    assert format_cost(0) == "$0.00"
    assert format_cost(0.0123) == "$0.0123"
    assert format_cost(2.5) == "$2.50"
