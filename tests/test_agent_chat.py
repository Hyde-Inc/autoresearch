import io
import json
from pathlib import Path

from rich.console import Console

from autoresearch.agent_chat import RoundChat, build_snapshot, recent_agent_events
from autoresearch.agent_status import EDITING, AgentStatus
from autoresearch.models import Idea


def _console() -> Console:
    return Console(file=io.StringIO(), width=100)


def _idea(title: str = "Global XGBoost") -> Idea:
    return Idea(
        title=title,
        hypothesis="Lag features beat repeating last week",
        instructions="Implement it",
        category="model",
        skills_used=[],
    )


def _tool_event(tool: str, file_path: str) -> str:
    return json.dumps(
        {
            "type": "tool_use",
            "part": {"tool": tool, "state": {"input": {"filePath": file_path}, "title": ""}},
        }
    )


def test_recent_agent_events_reads_log_tail(tmp_path: Path) -> None:
    log = tmp_path / "agent.jsonl"
    lines = [
        "not json",
        _tool_event("read", "solution/train.py"),
        _tool_event("edit", "solution/train.py"),
        _tool_event("edit", "solution/train.py"),  # duplicate collapses
        json.dumps({"type": "step_finish", "part": {}}),  # carries no action
    ]
    log.write_text("\n".join(lines) + "\n")
    events = recent_agent_events(log)
    assert events == ["read solution/train.py", "edit solution/train.py"]


def test_recent_agent_events_handles_missing_log(tmp_path: Path) -> None:
    assert recent_agent_events(None) == []
    assert recent_agent_events(tmp_path / "nope.jsonl") == []


def test_build_snapshot_includes_phase_hypothesis_and_trail(tmp_path: Path) -> None:
    status = AgentStatus(index=1, attempt_id="abc12345", title="Global XGBoost")
    status.set(EDITING, "edit solution/train.py")
    snapshot = build_snapshot("Round 1 · baseline wmape 0.10", [_idea()], [status])
    assert "Round 1 · baseline wmape 0.10" in snapshot
    assert "Agent 1 - Global XGBoost" in snapshot
    assert "phase: Editing" in snapshot
    assert "Lag features beat repeating last week" in snapshot
    assert "edit solution/train.py" in snapshot


def test_build_snapshot_prefers_log_events_over_trail(tmp_path: Path) -> None:
    log = tmp_path / "agent.jsonl"
    log.write_text(_tool_event("read", "seed/data/train.parquet") + "\n")
    status = AgentStatus(index=1, attempt_id="abc12345", title="Global XGBoost")
    status.log_path = log
    status.set(EDITING, "stale trail action")
    snapshot = build_snapshot("Round 1", [_idea()], [status])
    assert "- read data/train.parquet" in snapshot
    # The trail is superseded by log events in the bullet list; the stale trail
    # entry only survives as the row's current-action field.
    assert "- stale trail action" not in snapshot


class _FakeChat:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def turn(self, messages: list[dict], listener, tools=None) -> dict:
        self.calls.append([dict(m) for m in messages])
        listener.on_content("Agent 1 is editing train.py")
        return {"role": "assistant", "content": "Agent 1 is editing train.py"}


def test_round_chat_ask_sends_fresh_snapshot_and_keeps_history() -> None:
    status = AgentStatus(index=1, attempt_id="abc12345", title="Global XGBoost")
    chat = RoundChat.__new__(RoundChat)
    chat.chat = _FakeChat()
    chat.header = "Round 1"
    chat.ideas = [_idea()]
    chat.statuses = [status]
    chat.messages = [{"role": "system", "content": "system"}]

    answer = chat.ask("what is agent 1 doing?", _console())
    assert answer == "Agent 1 is editing train.py"
    sent = chat.chat.calls[0]
    assert sent[-1]["role"] == "user"
    assert "Live status snapshot:" in sent[-1]["content"]
    assert "what is agent 1 doing?" in sent[-1]["content"]
    assert chat.messages[-1] == {"role": "assistant", "content": "Agent 1 is editing train.py"}

    status.set(EDITING, "edit solution/features.py")
    chat.ask("and now?", _console())
    assert "edit solution/features.py" in chat.chat.calls[1][-1]["content"]


def test_round_chat_session_exits_on_empty_input(monkeypatch) -> None:
    chat = RoundChat.__new__(RoundChat)
    chat.header = "Round 1"
    chat.ideas = []
    chat.statuses = []
    chat.messages = []
    asked: list[str] = []
    chat.ask = lambda question, console: asked.append(question)  # type: ignore[method-assign]

    console = _console()
    replies = iter(["what is happening", ""])
    monkeypatch.setattr(console, "input", lambda prompt="": next(replies))
    chat.session(console)
    assert asked == ["what is happening"]


def test_round_chat_session_survives_ask_failure(monkeypatch) -> None:
    chat = RoundChat.__new__(RoundChat)
    chat.header = "Round 1"
    chat.ideas = []
    chat.statuses = []
    chat.messages = []

    def boom(question, console):
        raise RuntimeError("network down")

    chat.ask = boom  # type: ignore[method-assign]
    console = _console()
    replies = iter(["why so slow", "exit"])
    monkeypatch.setattr(console, "input", lambda prompt="": next(replies))
    chat.session(console)
    assert "chat failed: network down" in console.file.getvalue()
