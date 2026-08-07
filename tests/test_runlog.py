from autoresearch.runlog import format_agent_event


def test_formats_file_edit_event() -> None:
    event = {
        "type": "tool_use",
        "part": {
            "tool": "edit",
            "state": {
                "status": "completed",
                "input": {"filePath": "solution/train.py"},
            },
        },
    }
    assert format_agent_event(event) == "changed solution/train.py"


def test_formats_shell_and_text_events() -> None:
    shell = {
        "type": "tool_use",
        "part": {
            "tool": "bash",
            "state": {
                "status": "completed",
                "input": {"command": "uv run python solution/train.py"},
            },
        },
    }
    text = {"type": "text", "part": {"text": "The local backtest improved WMAPE."}}
    assert format_agent_event(shell) == "running a local model check"
    assert format_agent_event(text) == "research note: The local backtest improved WMAPE."


def test_ignores_low_signal_events() -> None:
    assert format_agent_event({"type": "step_start", "part": {}}) is None
    assert (
        format_agent_event(
            {
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {"status": "completed", "input": {"path": "TASK.md"}},
                },
            }
        )
        is None
    )


def test_formats_runtime_retry() -> None:
    assert (
        format_agent_event({"type": "runtime_retry", "attempt": 2, "max_attempts": 3})
        == "OpenCode startup failed, retrying (2/3)"
    )
    assert (
        format_agent_event({"type": "text", "part": {"text": "Let me explore the files."}})
        is None
    )
