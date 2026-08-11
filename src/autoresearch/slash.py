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

_APPROVALS = {
    "approve", "approved", "approve all", "go", "go ahead", "yes", "y", "ok", "okay",
    "start", "run", "run it", "launch", "proceed", "do it", "ship it", "lgtm",
    "looks good", "looks good to me", "sounds good", "execute", "execute it",
    "execute the plan", "run the plan",
}
_STOPS = {"stop", "quit", "exit", "end", "done", "no", "cancel", "abort"}


def parse_reply(text: str) -> str:
    """Classify a review reply: 'approve', 'stop', or 'revise' (free-form feedback).

    Whole-message approval/stop phrases decide; anything else is revision feedback.
    """
    normalized = " ".join(text.lower().replace("!", "").replace(".", "").split())
    if normalized in _APPROVALS:
        return "approve"
    if normalized in _STOPS:
        return "stop"
    return "revise"


@dataclass
class SessionSettings:
    goal: str | None = None
    baseline_path: str | None = None
    n_agents: int = 3
    metric: str = "wmape"
    metric_description: str | None = None
    guardrails: list[str] = field(default_factory=lambda: ["runtime_s<=600"])
    rounds: int = 4
    timeout_s: int = 1200
    budget_s: int = 3600

    def overrides(self) -> dict:
        """Settings in generated-task-config form for prepare_workspace."""
        return {
            "metric": {"name": self.metric, "direction": "min"},
            "guardrails": list(self.guardrails),
            "agents": {
                "count": self.n_agents,
                "timeout_s": self.timeout_s,
                "budget_s": self.budget_s,
            },
            "budget": {"rounds": self.rounds},
        }

    def describe(self) -> str:
        """Plain-text snapshot the director can read."""
        metric_line = (
            f"metric: CUSTOM (needs worked-example confirmation): {self.metric_description}"
            if self.metric_description
            else f"metric: {self.metric}"
        )
        return (
            f"goal: {self.goal or 'not set yet'}\n"
            f"baseline model: {self.baseline_path or 'not pinned; discover one in the repo'}\n"
            f"parallel researches per round (n_agents): {self.n_agents}\n"
            f"{metric_line}\n"
            f"guardrails: {self.guardrails}\n"
            f"max rounds: {self.rounds}\n"
            f"agent session timeout: {self.timeout_s}s\n"
            f"per-experiment budget: {self.budget_s}s"
        )


@dataclass
class SlashResult:
    handled: bool
    note: str | None = None
    """Authoritative settings note to forward to the director, if any."""


# (usage, description) for the /help table and the input box's completion menu.
COMMANDS = [
    ("/goal <text>", "What the research must improve, e.g. /goal reduce wmape"),
    ("/baseline <path>", "Pin an existing model script as the baseline, e.g. /baseline models/arima.py"),
    ("/n_agents <1-5>", "How many researches run in parallel each round"),
    ("/metric <name|description>", "wmape, mape, rmse, bias_pct - or describe a custom metric in plain English"),
    ("/guardrail <expr>", "Add a guardrail, e.g. /guardrail bias_pct within -8..8"),
    ("/rounds <n>", "Maximum research rounds"),
    ("/timeout <seconds>", "Coding agent timeout for one session of the fix loop"),
    ("/budget <seconds>", "Total per-experiment wall clock across all sessions"),
    ("/status", "Show the current settings"),
    ("/help", "Show this list"),
]


def print_help(console: Console) -> None:
    table = Table(title="Slash commands", show_header=False)
    table.add_column(style="bold blue")
    table.add_column()
    for command, description in COMMANDS:
        table.add_row(command, description)
    console.print(table)


def print_status(console: Console, settings: SessionSettings) -> None:
    table = Table(title="Current settings", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("goal", settings.goal or "[dim]not set[/dim]")
    table.add_row("baseline", settings.baseline_path or "[dim]not pinned[/dim]")
    table.add_row("n_agents", str(settings.n_agents))
    table.add_row(
        "metric",
        f"custom (unconfirmed): {settings.metric_description}"
        if settings.metric_description
        else settings.metric,
    )
    table.add_row("guardrails", "\n".join(settings.guardrails) or "-")
    table.add_row("rounds", str(settings.rounds))
    table.add_row("timeout", f"{settings.timeout_s}s")
    table.add_row("budget", f"{settings.budget_s}s")
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
            if not argument:
                raise ValueError(
                    "usage: /metric <wmape|mape|rmse|bias_pct> or a plain-English description"
                )
            if argument in {"wmape", "mape", "rmse", "bias_pct"}:
                settings.metric = argument
                settings.metric_description = None
                console.print(f"[green]Metric set:[/green] {argument}")
                return SlashResult(True, f"primary metric set to {argument}")
            settings.metric_description = argument
            console.print(f"[green]Custom metric requested:[/green] {argument}")
            return SlashResult(
                True,
                "the user wants a CUSTOM evaluation metric described as: "
                f'"{argument}". Before locking in, call define_custom_metric with this '
                "description, present the worked example and verification checks, and get an "
                "explicit yes (re-run it if they want changes). Then during lock-in call "
                "adopt_custom_metric so every experiment is scored with it. Do not call "
                "write_plan until it is adopted.",
            )
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
            console.print(f"[green]Agent session timeout:[/green] {settings.timeout_s}s")
            return SlashResult(True, f"agent session timeout set to {settings.timeout_s}s")
        if command == "budget":
            settings.budget_s = _positive_int(argument, "budget", 60)
            console.print(f"[green]Per-experiment budget:[/green] {settings.budget_s}s")
            return SlashResult(True, f"per-experiment budget set to {settings.budget_s}s")
        console.print(f"[dark_orange3]Unknown command /{command}.[/dark_orange3]")
        print_help(console)
        return SlashResult(handled=True)
    except ValueError as exc:
        console.print(f"[dark_orange3]! {exc}[/dark_orange3]")
        return SlashResult(handled=True)
