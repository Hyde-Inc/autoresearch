"""Live terminal dashboard for a round of parallel research agents.

While agents run, one Rich ``Live`` owns the screen: a table with a row per
experiment (phase, current action, elapsed time), a detail pane showing the
selected agent's recent actions and log path, and a one-line key legend. A
raw-key listener is active only while the dashboard is up: digits and arrow
keys select a row, ``x`` cancels the selected agent, ``q`` cancels the whole
round, and ``?`` toggles help. Terminal settings are restored on every exit
path. Outside a real terminal (CI, piped output) the dashboard degrades to
periodic plain-text summaries and no key listener.
"""

from __future__ import annotations

import contextlib
import os
import select
import sys
import threading
from collections.abc import Callable
from typing import Self

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .agent_status import CANCELLED, FAILED, PASSED, REJECTED, TRAINING, AgentStatus

_PHASE_STYLES = {
    PASSED: "bold green",
    FAILED: "bold red",
    REJECTED: "yellow",
    CANCELLED: "dim",
    TRAINING: "blue",
}

_LEGEND = "[1-9/↑↓] select   [c] chat   [x] cancel selected   [q] stop round   [?] help"
_HELP_LINES = (
    "1-9 or ↑/↓  select an agent row",
    "c           chat with the director about the running agents",
    "x           cancel the selected agent (others keep running)",
    "q           cancel every agent and stop the research run",
    "?           hide this help",
    "Logs        each agent streams to its .jsonl log (path shown above)",
)


class AgentDashboard:
    """Single owner of the terminal during parallel execution."""

    def __init__(
        self,
        console: Console,
        header: str,
        agents: list[AgentStatus],
        *,
        on_cancel_agent: Callable[[int], None] | None = None,
        on_cancel_round: Callable[[], None] | None = None,
        on_chat: Callable[[], None] | None = None,
        summary_interval_s: float = 20.0,
    ) -> None:
        self.console = console
        self.header = header
        self.agents = agents
        self.on_cancel_agent = on_cancel_agent
        self.on_cancel_round = on_cancel_round
        self.on_chat = on_chat
        self.summary_interval_s = summary_interval_s
        self.selected = 0
        self.show_help = False
        self._stop = threading.Event()
        self._live: Live | None = None
        self._key_thread: threading.Thread | None = None
        self._summary_thread: threading.Thread | None = None
        self._stdin_fd: int | None = None
        self._saved_termios: object | None = None

    # ------------------------------------------------------------- rendering

    def render(self) -> RenderableType:
        # Expand to exactly the terminal width and let the two text-heavy
        # columns share the remaining space. The previous fixed maxima could
        # produce a table wider than a narrow Cursor terminal; terminal-level
        # wrapping then made Rich miscount rows and leave old Live frames
        # behind on screen.
        table = Table(box=None, pad_edge=False, expand=True)
        table.add_column("#", justify="right", width=3)
        table.add_column(
            "Experiment", min_width=16, ratio=3, no_wrap=True, overflow="ellipsis"
        )
        table.add_column("Phase", width=14, no_wrap=True, overflow="ellipsis")
        table.add_column(
            "Current action", min_width=16, ratio=4, no_wrap=True, overflow="ellipsis"
        )
        table.add_column("Elapsed", justify="right", width=7, no_wrap=True)
        for position, agent in enumerate(self.agents):
            marker = ">" if position == self.selected else " "
            style = _PHASE_STYLES.get(agent.phase, "cyan")
            table.add_row(
                f"{marker}{agent.index}",
                agent.title,
                Text(agent.phase, style=style),
                Text(agent.action, style="dim" if agent.done else ""),
                agent.elapsed,
            )
        parts: list[RenderableType] = [Text(self.header, style="bold"), Text(), table, Text()]
        selected = self.agents[self.selected] if self.agents else None
        if selected is not None:
            trail = " → ".join(selected.trail) or "waiting for the first event"
            parts.append(Text(f"Selected: {trail}", style="italic"))
            if selected.log_path is not None:
                parts.append(Text(f"Log: {selected.log_path}", style="dim"))
        if self.show_help:
            parts.append(Text())
            parts.extend(Text(line, style="dim") for line in _HELP_LINES)
        else:
            parts.append(Text(_LEGEND, style="dim"))
        return Group(*parts)

    def _summary_line(self) -> str:
        chunks = [
            f"{agent.index} {agent.title[:28]}: {agent.phase} ({agent.action}, {agent.elapsed})"
            for agent in self.agents
        ]
        return "[dashboard] " + " | ".join(chunks)

    # ------------------------------------------------------------------ keys

    def handle_key(self, key: str) -> None:
        if not self.agents:
            return
        if key == "?":
            self.show_help = not self.show_help
        elif key in ("up", "k"):
            self.selected = max(0, self.selected - 1)
        elif key in ("down", "j"):
            self.selected = min(len(self.agents) - 1, self.selected + 1)
        elif key.isdigit() and 1 <= int(key) <= len(self.agents):
            self.selected = int(key) - 1
        elif key == "x" and self.on_cancel_agent is not None:
            self.on_cancel_agent(self.selected)
        elif key == "q" and self.on_cancel_round is not None:
            self.on_cancel_round()
        elif key == "c" and self.on_chat is not None:
            self._chat()

    def _chat(self) -> None:
        """Suspend the live table, hand the terminal to the chat, then resume.

        Runs on the key-listener thread; the agents' asyncio tasks are
        unaffected. The terminal leaves cbreak so the chat gets normal line
        input, and returns to cbreak afterwards.
        """
        assert self.on_chat is not None
        self._suspend_live()
        self._restore_cooked_terminal()
        try:
            self.on_chat()
        finally:
            self._enter_cbreak()
            self._resume_live()

    def _suspend_live(self) -> None:
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.stop()
            self._live = None

    def _resume_live(self) -> None:
        if self.console.is_terminal and not self._stop.is_set():
            self._live = Live(
                get_renderable=self.render,
                console=self.console,
                refresh_per_second=6,
                transient=False,
            )
            with contextlib.suppress(Exception):
                self._live.start()

    def _restore_cooked_terminal(self) -> None:
        if self._stdin_fd is None or self._saved_termios is None:
            return
        with contextlib.suppress(Exception):
            import termios

            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._saved_termios)

    def _enter_cbreak(self) -> None:
        if self._stdin_fd is None:
            return
        with contextlib.suppress(Exception):
            import tty

            tty.setcbreak(self._stdin_fd)

    def _read_keys(self) -> None:
        """Read single keys from the real terminal; restore settings no matter what."""
        try:
            import termios
            import tty
        except ImportError:  # pragma: no cover - non-POSIX platform
            return
        fd = sys.stdin.fileno()
        try:
            saved = termios.tcgetattr(fd)
        except (termios.error, ValueError, OSError):
            return
        self._stdin_fd = fd
        self._saved_termios = saved
        try:
            tty.setcbreak(fd)  # cbreak keeps ISIG, so Ctrl-C still interrupts
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(fd, 1)
                if not data:
                    break
                key = data.decode(errors="ignore")
                if key == "\x1b":
                    ready, _, _ = select.select([fd], [], [], 0.05)
                    sequence = os.read(fd, 2).decode(errors="ignore") if ready else ""
                    key = {"[A": "up", "[B": "down"}.get(sequence, "")
                if key:
                    with contextlib.suppress(Exception):
                        self.handle_key(key)
        finally:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    # --------------------------------------------------------- lifecycle

    def _print_summary(self) -> None:
        self.console.print(self._summary_line(), markup=False, highlight=False)

    def _print_summaries(self) -> None:
        while not self._stop.wait(self.summary_interval_s):
            self._print_summary()

    def __enter__(self) -> Self:
        if self.console.is_terminal:
            self._live = Live(
                get_renderable=self.render,
                console=self.console,
                refresh_per_second=6,
                transient=False,
            )
            self._live.start()
            if sys.stdin.isatty():
                self._key_thread = threading.Thread(
                    target=self._read_keys, name="dashboard-keys", daemon=True
                )
                self._key_thread.start()
        else:
            self._print_summary()
            self._summary_thread = threading.Thread(
                target=self._print_summaries, name="dashboard-summary", daemon=True
            )
            self._summary_thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._key_thread is not None:
            self._key_thread.join(timeout=1.0)
        if self._summary_thread is not None:
            self._summary_thread.join(timeout=self.summary_interval_s + 0.5)
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.stop()
        else:
            self._print_summary()
