"""Rendering, key handling, and fallback behavior of the parallel-agent dashboard."""

import io
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from rich.console import Console

from autoresearch.agent_status import EVALUATING, PASSED, AgentStatus
from autoresearch.dashboard import AgentDashboard


def _agents() -> list[AgentStatus]:
    first = AgentStatus(index=1, attempt_id="aaa11111", title="Global XGBoost")
    first.set("Editing", "write solution/train.py")
    first.log_path = Path("/runs/logs/aaa11111.jsonl")
    second = AgentStatus(index=2, attempt_id="bbb22222", title="Intermittent-demand route")
    second.set(EVALUATING, "validation split")
    third = AgentStatus(index=3, attempt_id="ccc33333", title="ARIMA + promo")
    third.finish(PASSED, "wmape 0.0981")
    return [first, second, third]


def _rendered(dashboard: AgentDashboard) -> str:
    # Explicit height too: rich only honors an explicit width unconditionally
    # (e.g. under TERM=dumb in CI) when both dimensions are pinned.
    console = Console(width=140, height=50, file=io.StringIO(), legacy_windows=False)
    with console.capture() as capture:
        console.print(dashboard.render())
    return capture.get()


def test_render_shows_rows_selection_detail_and_legend() -> None:
    console = Console(file=io.StringIO())
    dashboard = AgentDashboard(console, "Round 1 · baseline WMAPE 0.1039 · 3 agents", _agents())
    text = _rendered(dashboard)
    assert "Round 1 · baseline WMAPE 0.1039 · 3 agents" in text
    assert "Global XGBoost" in text
    assert "Intermittent-demand" in text
    assert "Protected eval" in text
    assert "Passed" in text
    assert ">1" in text  # selection marker on the first row
    assert "Selected: write solution/train.py" in text
    assert "aaa11111.jsonl" in text
    assert "[x] cancel selected" in text


def test_render_stays_within_narrow_cursor_terminal_width() -> None:
    buffer = io.StringIO()
    console = Console(width=100, height=50, file=buffer, legacy_windows=False)
    dashboard = AgentDashboard(
        console,
        "Round 1 · baseline WMAPE 0.1039 · 3 agents",
        _agents(),
    )
    console.print(dashboard.render())
    lines = buffer.getvalue().splitlines()
    assert all(len(line) <= 100 for line in lines)
    assert any("Experiment" in line and "Elapsed" in line for line in lines)
    assert sum("Global XGBoost" in line for line in lines) == 1
    assert sum("Intermittent-demand" in line for line in lines) == 1
    assert sum("ARIMA + promo" in line for line in lines) == 1


def test_keys_select_rows_toggle_help_and_fire_callbacks() -> None:
    cancelled: list[int] = []
    stopped: list[bool] = []
    console = Console(file=io.StringIO())
    dashboard = AgentDashboard(
        console,
        "Round 1",
        _agents(),
        on_cancel_agent=cancelled.append,
        on_cancel_round=lambda: stopped.append(True),
    )
    dashboard.handle_key("3")
    assert dashboard.selected == 2
    dashboard.handle_key("up")
    assert dashboard.selected == 1
    dashboard.handle_key("down")
    dashboard.handle_key("down")  # clamped at the last row
    assert dashboard.selected == 2
    dashboard.handle_key("9")  # out-of-range digit is ignored
    assert dashboard.selected == 2
    dashboard.handle_key("x")
    assert cancelled == [2]
    dashboard.handle_key("q")
    assert stopped == [True]
    dashboard.handle_key("?")
    assert dashboard.show_help
    assert "cancel the selected agent" in _rendered(dashboard)
    dashboard.handle_key("?")
    assert not dashboard.show_help


def test_chat_key_suspends_live_and_fires_callback() -> None:
    chats: list[bool] = []
    console = Console(file=io.StringIO())
    dashboard = AgentDashboard(console, "Round 1", _agents(), on_chat=lambda: chats.append(True))
    dashboard.handle_key("c")
    assert chats == [True]
    # Without a chat callback the key is a no-op.
    silent = AgentDashboard(console, "Round 1", _agents())
    silent.handle_key("c")  # must not raise
    assert "[c] chat" in _rendered(dashboard)


def test_non_tty_fallback_prints_periodic_plain_summaries() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=200)
    dashboard = AgentDashboard(console, "Round 1", _agents(), summary_interval_s=0.05)
    with dashboard:
        assert dashboard._live is None
        assert dashboard._key_thread is None
        time.sleep(0.18)
    output = buffer.getvalue()
    assert output.count("[dashboard]") >= 3  # initial + periodic + final
    assert "Global XGBoost" in output


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="requires a POSIX pty")
def test_key_reader_handles_keys_and_restores_terminal_state() -> None:
    import termios

    controller, follower = os.openpty()
    saved_stdin = sys.stdin
    seen: list[str] = []
    try:
        sys.stdin = os.fdopen(follower, "r", closefd=False)
        before = termios.tcgetattr(follower)
        console = Console(file=io.StringIO())
        dashboard = AgentDashboard(console, "Round 1", _agents())
        dashboard.handle_key = seen.append  # type: ignore[method-assign]
        reader = threading.Thread(target=dashboard._read_keys, daemon=True)
        reader.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:  # wait until the reader entered cbreak mode
            if not termios.tcgetattr(follower)[3] & termios.ICANON:
                break
            time.sleep(0.01)
        os.write(controller, b"2")
        os.write(controller, b"\x1b[A")  # up arrow
        os.write(controller, b"x")
        deadline = time.monotonic() + 2.0
        while len(seen) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        dashboard._stop.set()
        reader.join(timeout=2.0)
        assert seen == ["2", "up", "x"]

        # cbreak mode was undone; PENDIN is transient kernel state (set when
        # type-ahead input existed during the mode switch), not ours to restore.
        def normalized(attrs: list) -> list:
            attrs = list(attrs)
            attrs[3] &= ~termios.PENDIN
            return attrs

        assert normalized(termios.tcgetattr(follower)) == normalized(before)
    finally:
        sys.stdin = saved_stdin
        os.close(controller)
        os.close(follower)
