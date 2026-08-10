"""Event classification and per-agent state for the parallel dashboard.

Fixtures mirror real `opencode run --format json` output: one JSON object per
line with a top-level `type` and a `part` payload.
"""

import time

from autoresearch.agent_status import (
    CANCELLED,
    EDITING,
    EVALUATING,
    EXPLORING,
    MAX_ACTION_CHARS,
    PASSED,
    TESTING,
    AgentStatus,
    classify_event,
)


def _tool_use(tool: str, tool_input: dict, title: str = "") -> dict:
    return {
        "type": "tool_use",
        "timestamp": 1767036061199,
        "sessionID": "ses_494719016ffe85dkDMj0FPRbHK",
        "part": {
            "id": "prt_b6b8e85bb001CzBoN2dDlEZJnP",
            "type": "tool",
            "callID": "call_bQWsNLvOrJGIOz",
            "tool": tool,
            "state": {
                "status": "completed",
                "input": tool_input,
                "output": "ok",
                "title": title,
            },
        },
    }


def test_read_and_grep_classify_as_exploring() -> None:
    phase, action = classify_event(_tool_use("read", {"filePath": "/w/t/solution/train.py"}))
    assert phase == EXPLORING
    assert action == "read solution/train.py"
    phase, action = classify_event(_tool_use("grep", {"pattern": "wmape"}))
    assert phase == EXPLORING
    assert "grep" in action


def test_write_and_edit_classify_as_editing() -> None:
    phase, action = classify_event(_tool_use("write", {"filePath": "solution/train.py"}))
    assert phase == EDITING
    assert action == "write solution/train.py"
    phase, _ = classify_event(_tool_use("edit", {"filePath": "solution/features.py"}))
    assert phase == EDITING


def test_test_like_bash_classifies_as_testing_other_bash_keeps_phase() -> None:
    phase, action = classify_event(
        _tool_use("bash", {"command": "uv run python solution/train.py"}, "Run local backtest")
    )
    assert phase == TESTING
    assert action == "Run local backtest"
    phase, action = classify_event(_tool_use("bash", {"command": "ls data/"}))
    assert phase is None
    assert action == "ls data/"


def test_text_and_reasoning_become_thinking_actions() -> None:
    event = {"type": "text", "part": {"type": "text", "text": "I will add lag features."}}
    phase, action = classify_event(event)
    assert phase is None
    assert action == "thinking: I will add lag features."
    phase, action = classify_event({"type": "reasoning", "part": {"text": "check seasonality"}})
    assert phase is None
    assert action.startswith("thinking:")


def test_step_boundaries_and_malformed_events_are_ignored() -> None:
    assert classify_event({"type": "step_start", "part": {"type": "step-start"}}) == (None, None)
    assert classify_event(
        {"type": "step_finish", "part": {"type": "step-finish", "tokens": {"input": 10}}}
    ) == (None, None)
    assert classify_event("not json at all") == (None, None)
    assert classify_event({"type": "tool_use", "part": "corrupted"}) == (None, None)
    assert classify_event({}) == (None, None)


def test_error_events_surface_the_message() -> None:
    event = {"type": "error", "error": {"name": "APIError", "data": {"message": "rate limited"}}}
    phase, action = classify_event(event)
    assert phase is None
    assert action == "error: rate limited"


def test_long_details_are_truncated_safely() -> None:
    huge = "x" * 5000
    _, action = classify_event(_tool_use("bash", {"command": f"pytest {huge}"}))
    assert len(action) <= MAX_ACTION_CHARS
    assert action.endswith("…")


def test_status_tracks_trail_and_freezes_after_terminal_phase() -> None:
    status = AgentStatus(index=1, attempt_id="abc12345", title="Global XGBoost")
    status.apply_event(_tool_use("read", {"filePath": "solution/train.py"}))
    status.apply_event(_tool_use("write", {"filePath": "solution/train.py"}))
    assert status.phase == EDITING
    assert list(status.trail) == ["read solution/train.py", "write solution/train.py"]
    status.set(EVALUATING, "validation split")
    status.finish(PASSED, "wmape 0.0981")
    frozen = status.elapsed_s
    assert status.done
    # Late events and duplicate finishes must not disturb the final state.
    status.apply_event(_tool_use("write", {"filePath": "solution/other.py"}))
    status.finish(CANCELLED)
    time.sleep(0.02)
    assert status.phase == PASSED
    assert status.elapsed_s == frozen
    assert status.elapsed.count(":") == 1
