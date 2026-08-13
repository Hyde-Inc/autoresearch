import io
from types import SimpleNamespace

from prompt_toolkit.document import Document
from rich.console import Console

from autoresearch.chat import (
    SlashCompleter,
    TurnRenderer,
    _accept_selected_completion,
    accumulate_stream,
)
from autoresearch.slash import COMMANDS


class Recorder:
    def __init__(self) -> None:
        self.reasoning: list[str] = []
        self.content: list[str] = []

    def on_reasoning(self, token: str) -> None:
        self.reasoning.append(token)

    def on_content(self, token: str) -> None:
        self.content.append(token)


def _chunk(**delta_fields) -> SimpleNamespace:
    fields = {"content": None, "reasoning": None, "tool_calls": None, **delta_fields}
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(**fields))])


def _tool_delta(index: int, call_id: str | None, name: str | None, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_accumulate_stream_collects_reasoning_content_and_tool_calls() -> None:
    listener = Recorder()
    chunks = [
        _chunk(reasoning="thinking about "),
        _chunk(reasoning="the goal"),
        _chunk(content="Here is "),
        _chunk(content="the plan."),
        _chunk(tool_calls=[_tool_delta(0, "call_1", "start_research", '{"ideas": ')]),
        _chunk(tool_calls=[_tool_delta(0, None, None, "[]}")]),
    ]
    message = accumulate_stream(chunks, listener)
    assert "".join(listener.reasoning) == "thinking about the goal"
    assert message["content"] == "Here is the plan."
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "start_research", "arguments": '{"ideas": []}'},
        }
    ]


def test_accumulate_stream_without_tools_or_content() -> None:
    listener = Recorder()
    message = accumulate_stream([_chunk(reasoning="hmm")], listener)
    assert message == {"role": "assistant", "content": None}


def _completions(text: str) -> list[str]:
    completer = SlashCompleter(COMMANDS)
    return [item.text for item in completer.get_completions(Document(text, len(text)), None)]


def test_typing_slash_pops_the_full_command_menu() -> None:
    assert _completions("/") == [usage.split()[0] for usage, _ in COMMANDS]


def test_menu_filters_as_you_type_and_stays_out_of_plain_text() -> None:
    assert _completions("/g") == ["/goal", "/guardrail"]
    assert _completions("/GOAL") == ["/goal"]
    assert _completions("reduce wmape") == []  # plain chat: no menu
    assert _completions("/goal reduce") == []  # argument typing: menu closed


def test_enter_accepts_selected_completion_without_submitting_prefix() -> None:
    applied = []
    inserted = []
    completion = SimpleNamespace(text="/metric")
    buffer = SimpleNamespace(
        complete_state=SimpleNamespace(current_completion=completion),
        apply_completion=applied.append,
        insert_text=inserted.append,
    )
    _accept_selected_completion(SimpleNamespace(current_buffer=buffer))
    assert applied == [completion]
    assert inserted == [" "]


def test_turn_renderer_renders_markdown_and_wraps_complete_words() -> None:
    output = io.StringIO()
    # force_terminal=False: rich flips into terminal mode when FORCE_COLOR or
    # GITHUB_ACTIONS is in the environment, which changes where Live wraps lines.
    # Pinning it keeps the rendered output identical on dev machines and CI.
    renderer = TurnRenderer(Console(file=output, width=45, color_system=None, force_terminal=False))
    renderer.on_content(
        "Baseline **WMAPE** is **0.104**. This sentence should wrap cleanly "
        "between words instead of splitting them.\n\n1. **Global XGBoost**"
    )
    renderer.finish()
    rendered = "\n".join(line.rstrip() for line in output.getvalue().splitlines())
    assert "**" not in rendered
    assert "Baseline WMAPE is 0.104" in rendered
    assert "1 Global XGBoost" in rendered
    assert "instead of\nsplitting them." in rendered
