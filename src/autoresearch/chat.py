"""Claude Code-style terminal chat: streamed reasoning, streamed replies, input box.

``stream_completion`` consumes an OpenAI-compatible streaming response and fans
tokens out to a renderer while accumulating the full assistant message,
including tool calls, so the conversation history stays valid.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from openai import BadRequestError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import completion_is_selected, has_completions
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


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

    def __init__(
        self,
        console: Console,
        speaker: str = "Research Director",
        on_first_token: Callable[[], None] | None = None,
    ):
        self.console = console
        self.speaker = speaker
        self.mode: str | None = None
        self.on_first_token = on_first_token
        self._started = False
        self._content_parts: list[str] = []
        self._live: Live | None = None

    def _start(self) -> None:
        if not self._started:
            self._started = True
            if self.on_first_token is not None:
                self.on_first_token()

    def on_reasoning(self, token: str) -> None:
        self._start()
        self._stop_live()
        if self.mode != "reasoning":
            self.console.print("\n[dim italic]* thinking[/dim italic]")
            self.mode = "reasoning"
        self.console.print(
            token, end="", style="dim", markup=False, highlight=False, soft_wrap=True
        )

    def on_content(self, token: str) -> None:
        self._start()
        if self.mode != "content":
            self.console.print(f"\n\n[bold blue]{self.speaker}[/bold blue]")
            self.mode = "content"
            self._live = Live(
                Markdown(""),
                console=self.console,
                refresh_per_second=12,
                vertical_overflow="visible",
            )
            self._live.start()
        self._content_parts.append(token)
        assert self._live is not None
        self._live.update(Markdown("".join(self._content_parts)))

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def finish(self) -> None:
        self._stop_live()
        if self.mode == "reasoning":
            self.console.print()
        self.mode = None


class SlashCompleter(Completer):
    """Pops the command menu the moment the line starts with ``/``."""

    def __init__(self, commands: Sequence[tuple[str, str]]):
        self.commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for usage, description in self.commands:
            name = usage.split()[0]
            if name.startswith(text.lower()):
                yield Completion(
                    name,
                    start_position=-len(text),
                    display=usage,
                    display_meta=description,
                )


_PROMPT_STYLE = Style.from_dict(
    {
        "frame": "ansibrightblack",
        "completion-menu.completion": "bg:ansibrightblack ansiwhite",
        "completion-menu.completion.current": "bg:ansicyan ansiblack",
        "completion-menu.meta.completion": "bg:ansibrightblack ansiwhite",
        "completion-menu.meta.completion.current": "bg:ansicyan ansiblack",
    }
)
_HISTORY = InMemoryHistory()
_KEY_BINDINGS = KeyBindings()


@_KEY_BINDINGS.add("enter", filter=has_completions & completion_is_selected)
def _accept_selected_completion(event) -> None:
    """Enter accepts the highlighted command instead of submitting a prefix."""
    completion = event.current_buffer.complete_state.current_completion
    event.current_buffer.apply_completion(completion)
    event.current_buffer.insert_text(" ")


def _read_line(console: Console, completions: Sequence[tuple[str, str]] | None) -> str:
    if completions:
        try:
            session: PromptSession = PromptSession(
                history=_HISTORY,
                completer=SlashCompleter(completions),
                complete_while_typing=True,
                key_bindings=_KEY_BINDINGS,
                style=_PROMPT_STYLE,
            )
            return session.prompt([("class:frame", "\u2502 "), ("bold", "> ")])
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception:  # noqa: BLE001, S110 - no usable tty; fall back to plain input
            pass
    return console.input("[dim]\u2502[/dim] [bold]>[/bold] ")


def input_box(console: Console, completions: Sequence[tuple[str, str]] | None = None) -> str:
    """Bordered single-line prompt, Claude Code style, with a slash-command menu.

    Typing ``/`` pops a completion menu of every command (when *completions* are
    provided); arrows select and Enter accepts. Re-asks on empty input.
    """
    width = max(20, min(console.width, 100))
    while True:
        console.print()
        console.print("╭" + "─" * (width - 1), style="dim")
        try:
            value = _read_line(console, completions)
        finally:
            console.print("╰" + "─" * (width - 1), style="dim")
        if value.strip():
            return value.strip()
        console.print("[dim]Type a reply to continue.[/dim]")
