"""Claude Code-style terminal chat: streamed reasoning, streamed replies, input box.

``stream_completion`` consumes an OpenAI-compatible streaming response and fans
tokens out to a renderer while accumulating the full assistant message,
including tool calls, so the conversation history stays valid.
"""

from __future__ import annotations

from typing import Any, Protocol

from openai import BadRequestError
from rich.console import Console


class TurnListener(Protocol):
    def on_reasoning(self, token: str) -> None: ...

    def on_content(self, token: str) -> None: ...


def accumulate_stream(chunks: Any, listener: TurnListener) -> dict:
    """Fold streamed deltas into one OpenAI-format assistant message."""
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    for chunk in chunks:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning:
            listener.on_reasoning(reasoning)
        if delta.content:
            content_parts.append(delta.content)
            listener.on_content(delta.content)
        for part in delta.tool_calls or []:
            slot = tool_calls.setdefault(
                part.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if part.id:
                slot["id"] = part.id
            if part.function:
                if part.function.name:
                    slot["function"]["name"] = part.function.name
                if part.function.arguments:
                    slot["function"]["arguments"] += part.function.arguments
    message: dict = {"role": "assistant", "content": "".join(content_parts) or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return message


class StreamingChat:
    """Streams completions through OpenRouter, surfacing reasoning tokens when offered."""

    def __init__(self, client: Any, model: str, temperature: float):
        self.client = client
        self.model = model
        self.temperature = temperature
        self._reasoning_supported = True

    def turn(
        self,
        messages: list[dict],
        listener: TurnListener,
        tools: list[dict] | None = None,
    ) -> dict:
        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if self._reasoning_supported:
            kwargs["extra_body"] = {"reasoning": {"effort": "medium"}}
        try:
            stream = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            if not self._reasoning_supported:
                raise
            self._reasoning_supported = False
            kwargs.pop("extra_body", None)
            stream = self.client.chat.completions.create(**kwargs)
        return accumulate_stream(stream, listener)


class TurnRenderer:
    """Prints one assistant turn: dim thinking stream, then the reply stream."""

    def __init__(self, console: Console, speaker: str = "Research Director"):
        self.console = console
        self.speaker = speaker
        self.mode: str | None = None

    def on_reasoning(self, token: str) -> None:
        if self.mode != "reasoning":
            self.console.print("\n[dim italic]* thinking[/dim italic]")
            self.mode = "reasoning"
        self.console.print(
            token, end="", style="dim", markup=False, highlight=False, soft_wrap=True
        )

    def on_content(self, token: str) -> None:
        if self.mode != "content":
            self.console.print(f"\n\n[bold cyan]{self.speaker}[/bold cyan]")
            self.mode = "content"
        self.console.print(token, end="", markup=False, highlight=False, soft_wrap=True)

    def finish(self) -> None:
        if self.mode is not None:
            self.console.print()
        self.mode = None


def input_box(console: Console) -> str:
    """Bordered single-line prompt, Claude Code style. Re-asks on empty input."""
    width = max(20, min(console.width, 100))
    while True:
        console.print()
        console.print("╭" + "─" * (width - 1), style="dim")
        try:
            value = console.input("[dim]│[/dim] [bold]>[/bold] ")
        finally:
            console.print("╰" + "─" * (width - 1), style="dim")
        if value.strip():
            return value.strip()
        console.print("[dim]Type a reply to continue.[/dim]")
