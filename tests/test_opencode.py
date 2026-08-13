"""Streaming events and cooperative cancellation of the opencode wrapper."""

import asyncio
import json
import os
import stat
import time
from pathlib import Path

import pytest

from autoresearch.opencode import run_opencode

STEP_START = json.dumps(
    {"type": "step_start", "sessionID": "ses_test01", "part": {"type": "step-start"}}
)
TOOL_USE = json.dumps(
    {
        "type": "tool_use",
        "sessionID": "ses_test01",
        "part": {
            "type": "tool",
            "tool": "read",
            "state": {"status": "completed", "input": {"filePath": "solution/train.py"}},
        },
    }
)


def _fake_opencode(tmp_path: Path, script_body: str) -> Path:
    """Install a shell script that stands in for the real opencode binary."""
    script = tmp_path / "fake-opencode"
    script.write_text(f"#!/bin/sh\n{script_body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _wait_for_death(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_events_stream_to_callback_and_log(tmp_path: Path, monkeypatch) -> None:
    script = _fake_opencode(
        tmp_path,
        f"echo '{STEP_START}'\necho '{TOOL_USE}'\necho 'not json at all'\n",
    )
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))
    events: list[dict] = []
    log = tmp_path / "logs" / "agent.jsonl"
    result = asyncio.run(
        run_opencode(tmp_path, "prompt", "provider/model", 30, log, on_event=events.append)
    )
    assert result.returncode == 0
    assert result.session_id == "ses_test01"
    assert [event["type"] for event in events] == ["step_start", "tool_use"]
    assert "not json at all" in log.read_text()  # malformed lines are kept for forensics


def test_broken_event_callback_does_not_kill_the_agent(tmp_path: Path, monkeypatch) -> None:
    script = _fake_opencode(tmp_path, f"echo '{STEP_START}'\necho '{TOOL_USE}'\n")
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))

    def explode(event: dict) -> None:
        raise RuntimeError("display bug")

    log = tmp_path / "agent.jsonl"
    result = asyncio.run(
        run_opencode(tmp_path, "prompt", "provider/model", 30, log, on_event=explode)
    )
    assert result.returncode == 0
    assert result.error is None
    assert STEP_START in log.read_text()


def test_non_json_stderr_is_returned_as_a_useful_failure(tmp_path: Path, monkeypatch) -> None:
    script = _fake_opencode(
        tmp_path,
        "printf '\\033[91mError:\\033[0m Unexpected error\\n\\ndatabase is locked\\n'\nexit 1",
    )
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))
    result = asyncio.run(
        run_opencode(tmp_path, "prompt", "provider/model", 30, tmp_path / "agent.jsonl")
    )
    assert result.returncode == 1
    assert result.error is not None
    assert "database is locked" in result.error
    assert "\x1b" not in result.error


def test_parallel_agents_get_isolated_disposable_databases(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent workers must not share OpenCode's global SQLite database."""
    script = _fake_opencode(
        tmp_path,
        "echo \"$XDG_DATA_HOME\" > \"$PWD/data-home.txt\"\n"
        "echo \"$XDG_STATE_HOME\" > \"$PWD/state-home.txt\"\n"
        f"echo '{STEP_START}'\n",
    )
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))
    worktrees = [tmp_path / f"worker-{index}" for index in range(3)]
    for worktree in worktrees:
        worktree.mkdir()

    async def scenario():
        return await asyncio.gather(
            *[
                run_opencode(
                    worktree,
                    "prompt",
                    "provider/model",
                    30,
                    tmp_path / "logs" / f"{index}.jsonl",
                )
                for index, worktree in enumerate(worktrees)
            ]
        )

    results = asyncio.run(scenario())
    data_homes = [Path((worktree / "data-home.txt").read_text().strip()) for worktree in worktrees]
    state_homes = [
        Path((worktree / "state-home.txt").read_text().strip()) for worktree in worktrees
    ]
    assert all(result.returncode == 0 for result in results)
    assert len(set(data_homes)) == len(worktrees)
    assert len(set(state_homes)) == len(worktrees)
    assert all("autoresearch-opencode-" in str(path) for path in data_homes + state_homes)
    # Temporary database/state roots are removed when the one-shot agents exit.
    assert all(not path.exists() for path in data_homes + state_homes)


def test_cancellation_kills_the_agent_and_keeps_the_log(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "agent.pid"
    script = _fake_opencode(
        tmp_path,
        f"echo '{STEP_START}'\necho $$ > {pid_file}\nexec sleep 30",
    )
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))
    events: list[dict] = []
    log = tmp_path / "agent.jsonl"

    async def scenario() -> None:
        task = asyncio.create_task(
            run_opencode(tmp_path, "prompt", "provider/model", 300, log, on_event=events.append)
        )
        deadline = time.monotonic() + 5.0
        while not pid_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    pid = int(pid_file.read_text())
    assert _wait_for_death(pid), "opencode subprocess survived cancellation"
    # The partial JSONL log written before the interrupt is preserved.
    assert STEP_START in log.read_text()
    assert events and events[0]["type"] == "step_start"


def test_timeout_still_terminates_and_reports(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "agent.pid"
    script = _fake_opencode(tmp_path, f"echo $$ > {pid_file}\nexec sleep 30")
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))
    # 2s, not 1s: under full-suite load the spawn can take a beat, and the
    # process must live long enough to write its pid file before the kill.
    result = asyncio.run(
        run_opencode(tmp_path, "prompt", "provider/model", 2, tmp_path / "agent.jsonl")
    )
    assert result.error is not None and "timeout" in result.error
    assert _wait_for_death(int(pid_file.read_text()))


def test_resume_reuses_the_session_and_the_persistent_data_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """The worker loop resumes one conversation across calls: the second run
    must pass --session, keep the caller-owned XDG data root, and append to
    the same log."""
    script = _fake_opencode(
        tmp_path,
        f'printf \'%s\\n\' "$@" > "$PWD/args-$OPENCODE_CALL.txt"\n'
        f'echo "$XDG_DATA_HOME" > "$PWD/data-home-$OPENCODE_CALL.txt"\n'
        f"echo '{STEP_START}'\n",
    )
    monkeypatch.setattr("autoresearch.opencode.shutil.which", lambda _: str(script))
    data_dir = tmp_path / "session-store"
    log = tmp_path / "agent.jsonl"

    async def scenario():
        monkeypatch.setenv("OPENCODE_CALL", "1")
        first = await run_opencode(
            tmp_path, "build it", "provider/model", 30, log, data_dir=data_dir
        )
        monkeypatch.setenv("OPENCODE_CALL", "2")
        second = await run_opencode(
            tmp_path,
            "the evaluator failed, fix it",
            "provider/model",
            30,
            log,
            session_id=first.session_id,
            data_dir=data_dir,
        )
        return first, second

    first, second = asyncio.run(scenario())
    assert first.session_id == "ses_test01"
    assert second.returncode == 0
    first_args = (tmp_path / "args-1.txt").read_text().splitlines()
    second_args = (tmp_path / "args-2.txt").read_text().splitlines()
    assert "--title" in first_args and "--session" not in first_args
    assert "--session" in second_args and "--title" not in second_args
    assert second_args[second_args.index("--session") + 1] == "ses_test01"
    # The project directory is pinned; opencode must not walk up out of the
    # worktree and resolve an enclosing repository as the project root.
    for args in (first_args, second_args):
        assert args[args.index("--dir") + 1] == str(tmp_path)
    # Both calls ran against the same caller-owned session store, which survives.
    homes = {
        (tmp_path / f"data-home-{index}.txt").read_text().strip() for index in (1, 2)
    }
    assert homes == {str(data_dir / "data")}
    assert data_dir.exists()
    # The shared log accumulated both sessions instead of being truncated.
    assert log.read_text().count(STEP_START) == 2
