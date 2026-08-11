"""Chat with the Research Director while a round of agents is running.

The dashboard's ``c`` key suspends the live table and opens this chat. Each
question is answered from a fresh snapshot of the round: every agent's phase,
elapsed time, and recent actions (from its status trail plus the tail of its
streamed JSONL event log), alongside the experiment hypotheses. The agents
themselves are untouched - they keep running on the asyncio loop while the
chat happens on the dashboard's key-listener thread.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .agent_status import AgentStatus, classify_event
from .chat import StreamingChat, TurnRenderer
from .models import Idea

_SYSTEM_PROMPT = (
    "You are the Research Director of an autonomous ML research lab. A round of parallel "
    "coding agents is implementing experiments right now, and the user is watching the "
    "live dashboard and asking you what is happening. Answer from the status snapshot "
    "and recent agent events provided with each question. Be concise and factual: refer "
    "to agents by their number and experiment title, and ground every claim in the "
    "snapshot. If the snapshot does not show something, say you cannot see it rather "
    "than guessing. Experiments are scored only by the protected evaluator after they "
    "finish, so never promise results early."
)

MAX_LOG_EVENTS = 25
_MAX_LOG_LINES = 400


def recent_agent_events(log_path: Path | None, limit: int = MAX_LOG_EVENTS) -> list[str]:
    """Human-readable actions from the tail of one agent's JSONL event log."""
    if log_path is None or not log_path.exists():
        return []
    actions: list[str] = []
    for line in log_path.read_text(errors="ignore").splitlines()[-_MAX_LOG_LINES:]:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        _, action = classify_event(event)
        if action and (not actions or actions[-1] != action):
            actions.append(action)
    return actions[-limit:]


def build_snapshot(header: str, ideas: list[Idea], statuses: list[AgentStatus]) -> str:
    """One textual snapshot of the round, refreshed for every question."""
    parts = [header]
    hypotheses = {idea.title: idea.hypothesis for idea in ideas}
    for status in statuses:
        parts.append(
            f"\nAgent {status.index} - {status.title} "
            f"[phase: {status.phase}, elapsed: {status.elapsed}, "
            f"current action: {status.action}]"
        )
        hypothesis = hypotheses.get(status.title)
        if hypothesis:
            parts.append(f"  hypothesis: {hypothesis}")
        events = recent_agent_events(status.log_path)
        for action in events or status.trail:
            parts.append(f"  - {action}")
    return "\n".join(parts)


class RoundChat:
    """Question-and-answer session about a live round, with conversation memory."""

    def __init__(
        self,
        client: object,
        model: str,
        temperature: float,
        *,
        header: str,
        ideas: list[Idea],
        statuses: list[AgentStatus],
    ) -> None:
        self.chat = StreamingChat(client, model.removeprefix("openrouter/"), temperature)
        self.header = header
        self.ideas = ideas
        self.statuses = statuses
        self.messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    def ask(self, question: str, console: Console) -> str:
        snapshot = build_snapshot(self.header, self.ideas, self.statuses)
        self.messages.append(
            {
                "role": "user",
                "content": f"Live status snapshot:\n{snapshot}\n\nQuestion: {question}",
            }
        )
        renderer = TurnRenderer(console)
        message = self.chat.turn(self.messages, renderer)
        renderer.finish()
        content = message.get("content") or ""
        self.messages.append({"role": "assistant", "content": content})
        return content

    def session(self, console: Console) -> None:
        """Blocking chat loop; an empty reply (or Ctrl-C) returns to the dashboard."""
        console.print()
        console.print(
            "[bold blue]Round chat[/bold blue] - the agents keep running. "
            "Ask what is happening; press Enter on an empty line to go back."
        )
        while True:
            try:
                question = console.input("\n[bold]chat >[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not question or question.lower() in {"back", "exit", "quit"}:
                return
            try:
                self.ask(question, console)
            except Exception as exc:  # noqa: BLE001 - a chat failure must not kill the round
                console.print(f"[red]chat failed: {exc}[/red]")
            console.print()
