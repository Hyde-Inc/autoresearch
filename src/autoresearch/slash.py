"""Slash commands: edit run settings inline in the chat, Cursor/Claude style.

Typing ``/goal reduce wmape`` or ``/n_agents 3`` in the input box updates the
session settings locally (no LLM round-trip). The director receives the change
as an authoritative note on the next message. There is no YAML to write:
defaults come from code and slash commands override them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

MIN_PARALLEL = 1
MAX_PARALLEL = 5


@dataclass
class SessionSettings:
    goal: str | None = None
    baseline_path: str | None = None
    n_agents: int = 3
    metric: str = "wmape"
    guardrails: list[str] = field(default_factory=lambda: ["runtime_s<=600"])
    rounds: int = 4
    timeout_s: int = 900

    def overrides(self) -> dict:
        """Settings in generated-task-config form for prepare_workspace."""
        return {
            "metric": {"name": self.metric, "direction": "min"},
            "guardrails": list(self.guardrails),
            "agents": {"count": self.n_agents, "timeout_s": self.timeout_s},
            "budget": {"rounds": self.rounds},
        }

    def describe(self) -> str:
        """Plain-text snapshot the director can read."""
        return (
            f"goal: {self.goal or 'not set yet'}\n"
            f"baseline model: {self.baseline_path or 'not pinned; discover one in the repo'}\n"
            f"parallel researches per round (n_agents): {self.n_agents}\n"
            f"metric: {self.metric}\n"
            f"guardrails: {self.guardrails}\n"
            f"max rounds: {self.rounds}\n"
            f"agent timeout: {self.timeout_s}s"
        )


@dataclass
class SlashResult:
    handled: bool
    note: str | None = None
    """Authoritative settings note to forward to the director, if any."""


_HELP_ROWS = [
    ("/goal <text>", "What the research must improve, e.g. /goal reduce wmape"),
    ("/baseline <path>", "Pin an existing model script as the baseline, e.g. /baseline models/arima.py"),
    ("/n_agents <1-5>", "How many researches run in parallel each round"),
    ("/metric <name>", "Primary metric: wmape, mape, rmse, or bias_pct"),
    ("/guardrail <expr>", "Add a guardrail, e.g. /guardrail bias_pct within -8..8"),
    ("/rounds <n>", "Maximum research rounds"),
    ("/timeout <seconds>", "Per-experiment coding agent timeout"),
    ("/status", "Show the current settings"),
    ("/help", "Show this list"),
]


def print_help(console: Console) -> None:
    table = Table(title="Slash commands", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column()
    for command, description in _HELP_ROWS:
        table.add_row(command, description)
    console.print(table)


def print_status(console: Console, settings: SessionSettings) -> None:
    table = Table(title="Current settings", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("goal", settings.goal or "[dim]not set[/dim]")
    table.add_row("baseline", settings.baseline_path or "[dim]not pinned[/dim]")
    table.add_row("n_agents", str(settings.n_agents))
    table.add_row("metric", settings.metric)
    table.add_row("guardrails", "\n".join(settings.guardrails) or "-")
    table.add_row("rounds", str(settings.rounds))
    table.add_row("timeout", f"{settings.timeout_s}s")
    console.print(table)


def _positive_int(value: str, name: str, low: int, high: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if parsed < low or (high is not None and parsed > high):
        limit = f"between {low} and {high}" if high else f"at least {low}"
        raise ValueError(f"{name} must be {limit}, got {parsed}")
    return parsed


def handle_slash(text: str, settings: SessionSettings, console: Console) -> SlashResult:
    """Apply one slash command. Returns handled=False when text is not a command."""
    if not text.startswith("/"):
        return SlashResult(handled=False)
    command, _, argument = text[1:].partition(" ")
    command = command.strip().lower()
    argument = argument.strip()
    try:
        if command == "help" or command == "":
            print_help(console)
            return SlashResult(handled=True)
        if command == "status":
            print_status(console, settings)
            return SlashResult(handled=True)
        if command == "goal":
            if not argument:
                raise ValueError("usage: /goal <what to improve>")
            settings.goal = argument
            console.print(f"[green]Goal set:[/green] {argument}")
            return SlashResult(True, f"goal set to: {argument}")
        if command == "baseline":
            if not argument:
                raise ValueError("usage: /baseline <path to model script in the repo>")
            settings.baseline_path = argument
            console.print(f"[green]Baseline pinned:[/green] {argument}")
            return SlashResult(
                True,
                f"baseline model pinned to {argument}; port this script faithfully as "
                "the incumbent baseline",
            )
        if command == "n_agents":
            settings.n_agents = _positive_int(argument, "n_agents", MIN_PARALLEL, MAX_PARALLEL)
            console.print(f"[green]Parallel researches per round:[/green] {settings.n_agents}")
            return SlashResult(True, f"n_agents set to {settings.n_agents}")
        if command == "metric":
            if argument not in {"wmape", "mape", "rmse", "bias_pct"}:
                raise ValueError("metric must be one of: wmape, mape, rmse, bias_pct")
            settings.metric = argument
            console.print(f"[green]Metric set:[/green] {argument}")
            return SlashResult(True, f"primary metric set to {argument}")
        if command == "guardrail":
            if not argument:
                raise ValueError("usage: /guardrail <expression>, e.g. bias_pct within -8..8")
            settings.guardrails.append(argument)
            console.print(f"[green]Guardrail added:[/green] {argument}")
            return SlashResult(True, f"guardrail added: {argument}")
        if command == "rounds":
            settings.rounds = _positive_int(argument, "rounds", 1)
            console.print(f"[green]Max rounds:[/green] {settings.rounds}")
            return SlashResult(True, f"max rounds set to {settings.rounds}")
        if command == "timeout":
            settings.timeout_s = _positive_int(argument, "timeout", 60)
            console.print(f"[green]Agent timeout:[/green] {settings.timeout_s}s")
            return SlashResult(True, f"agent timeout set to {settings.timeout_s}s")
        console.print(f"[yellow]Unknown command /{command}.[/yellow]")
        print_help(console)
        return SlashResult(handled=True)
    except ValueError as exc:
        console.print(f"[yellow]! {exc}[/yellow]")
        return SlashResult(handled=True)
