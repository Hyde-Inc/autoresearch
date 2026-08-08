"""Conversational research setup: one streamed chat that configures and launches a run.

The Research Director talks freely (thinking tokens included) and acts through
tools: saving the brief, defining a verified metric, swapping the baseline,
running the protected evaluation, and finally starting research with the
approved plan. Invalid input (like a bad file path) comes back as a tool error
the director explains in conversation instead of crashing the CLI.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chat import StreamingChat, TurnRenderer, input_box
from .config import TaskConfig, load_config
from .metrics import MetricInterpreter, MetricSpec, MetricValidation, adopt_spec, eval_columns
from .models import Idea
from .orchestrator import validate_baseline
from .skills import ResearchSkill, load_skills

MAX_TOOL_ROUNDS = 12


class ResearchBrief(BaseModel):
    goal: str
    context: str
    guardrails: list[str] = Field(default_factory=list)
    idea_hints: list[str] = Field(default_factory=list)
    metric_name: str = "wmape"
    metric_description: str = ""


def data_profile(config: TaskConfig) -> dict:
    frame = pd.read_parquet(config.resolve(config.data.train))
    target = pd.to_numeric(frame[config.data.target_column], errors="coerce")
    dates = pd.to_datetime(frame[config.data.date_column])
    return {
        "rows": len(frame),
        "items": int(frame[config.data.id_column].nunique()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "distinct_dates": int(dates.nunique()),
        "zero_share": round(float((target.fillna(0) == 0).mean()), 4),
        "columns": list(frame.columns),
    }


def apply_brief(config: TaskConfig, brief: ResearchBrief) -> None:
    assert config.config_path is not None
    raw = yaml.safe_load(config.config_path.read_text())
    raw["goal"] = brief.goal
    raw["context"] = brief.context
    raw["guardrails"] = brief.guardrails
    raw["idea_hints"] = brief.idea_hints
    if brief.metric_name and not brief.metric_description:
        raw["metric"] = {"name": brief.metric_name, "direction": "min"}
    config.config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))


def replace_baseline(config: TaskConfig, source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"baseline script not found: {source}")
    destination = config.resolve(config.workspace.seed) / "solution" / "train.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def append_context(config: TaskConfig, text: str) -> None:
    assert config.config_path is not None
    raw = yaml.safe_load(config.config_path.read_text())
    existing = str(raw.get("context", "")).strip()
    raw["context"] = "\n".join(part for part in (existing, text.strip()) if part)
    config.config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))


def print_spec(console: Console, spec: MetricSpec, validation: MetricValidation) -> None:
    console.print(f"\n[bold]Metric:[/bold] {spec.name} ({spec.direction}imize)")
    console.print(f"[bold]Understanding:[/bold] {spec.understanding}")
    table = Table(title="Hand-worked example")
    columns = list(spec.example.rows[0])
    for column in columns:
        table.add_column(column)
    for row in spec.example.rows:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)
    for step in spec.example.steps:
        console.print(f"  - {step}")
    console.print(f"[bold]Hand-computed value:[/bold] {spec.example.value}")
    for check in validation.checks:
        color = "green" if check.passed else "red"
        console.print(f"[{color}]{check.name}[/{color}]: {check.detail}")
    for warning in validation.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_IDEA_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "hypothesis": {"type": "string"},
            "instructions": {"type": "string"},
            "category": {"type": "string"},
            "skills_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "hypothesis", "instructions"],
    },
}

TOOLS = [
    _tool(
        "finalize_brief",
        "Save the agreed research brief to task.yaml. Call only after the user has confirmed "
        "the brief in conversation.",
        {
            "goal": {"type": "string"},
            "context": {"type": "string", "description": "Business context for the agents"},
            "guardrails": {"type": "array", "items": {"type": "string"}},
            "idea_hints": {"type": "array", "items": {"type": "string"}},
            "metric_name": {"type": "string", "description": "wmape, mape, rmse, or bias_pct"},
        },
        ["goal", "context", "guardrails", "idea_hints", "metric_name"],
    ),
    _tool(
        "define_custom_metric",
        "Turn a plain-language metric description into verified grader code with a hand-worked "
        "example. Use only when standard metrics do not fit.",
        {"description": {"type": "string"}},
        ["description"],
    ),
    _tool(
        "adopt_custom_metric",
        "Persist the most recently defined custom metric as the grader. Call only after the "
        "user confirmed it.",
        {},
        [],
    ),
    _tool(
        "replace_baseline",
        "Copy the user's own training script over the seeded baseline. Only call when the user "
        "supplied a file path.",
        {"path": {"type": "string"}},
        ["path"],
    ),
    _tool(
        "evaluate_baseline",
        "Run the current baseline through the protected evaluator and return validation and "
        "hidden holdout metrics.",
        {},
        [],
    ),
    _tool(
        "record_context",
        "Append a useful business fact the user shared (for example current production "
        "performance) to the task context.",
        {"text": {"type": "string"}},
        ["text"],
    ),
    _tool(
        "start_research",
        "Launch the autonomous research run with the approved round-1 experiment plan. Call "
        "only after the user approved the plan.",
        {"ideas": _IDEA_SCHEMA},
        ["ideas"],
    ),
]

_TOOL_LABELS = {
    "finalize_brief": "saving the research brief to task.yaml",
    "define_custom_metric": "writing and verifying the custom metric",
    "adopt_custom_metric": "adopting the custom metric as the grader",
    "replace_baseline": "installing your baseline script",
    "evaluate_baseline": "running the protected baseline evaluation",
    "record_context": "recording that in the task context",
    "start_research": "preparing the research run",
}


class SetupSession:
    """Drives the streamed setup conversation and executes the director's tools."""

    def __init__(self, config_path: Path, client, console: Console, experiments: int):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.console = console
        self.experiments = experiments
        self.chat = StreamingChat(
            client,
            self.config.director.model.removeprefix("openrouter/"),
            self.config.director.temperature,
        )
        self.skills: list[ResearchSkill] = load_skills(self.config)
        self.pending_spec: MetricSpec | None = None
        self.final_ideas: list[Idea] | None = None

    def _reload(self) -> None:
        self.config = load_config(self.config_path)

    def system_prompt(self) -> str:
        seed = self.config.resolve(self.config.workspace.seed)
        baseline = seed / "solution" / "train.py"
        contract_path = seed / "TASK.md"
        contract = (
            contract_path.read_text() if contract_path.exists() else self.config.description
        )
        skill_text = "\n\n".join(
            f"### {skill.name}\nWhen to use: {skill.description}\n{skill.body}"
            for skill in self.skills
        )
        return (
            "You are the Research Director of an autonomous demand forecasting lab, setting up "
            "a research run together with the user in a terminal chat.\n\n"
            "Conversation style: short messages, plain language, one question at a time. Never "
            "offer yes/no menus or numbered options; just talk. Never ask for information "
            "already in this prompt. If the user gives an unclear or invalid answer, say what "
            "went wrong and ask again; never stop the session.\n\n"
            "Flow:\n"
            "1. Understand what the research should improve. Fill gaps with sensible defaults "
            "instead of interrogating the user. Present the brief (goal, business context, "
            "metric, guardrails, research directions) in plain text, and once the user agrees, "
            "call finalize_brief.\n"
            "2. Standard metrics are wmape, mape, rmse, bias_pct. Only if the user needs custom "
            "grading logic, call define_custom_metric, walk them through the verified result, "
            "and call adopt_custom_metric after they agree.\n"
            f"3. The seeded baseline is {baseline} (a seasonal-naive forecaster). Default to it. "
            "Only if the user offers their own script, ask for its path and call "
            "replace_baseline. Then call evaluate_baseline and interpret the numbers for the "
            "user in one or two sentences.\n"
            f"4. Propose a round-1 plan of exactly {self.experiments} experiment(s), each a "
            "single testable change, citing which skills informed it. Present the plan in plain "
            "text, and once the user approves, call start_research with the ideas (fields: "
            "title, hypothesis, instructions, category, skills_used).\n\n"
            "Use record_context whenever the user shares a useful business fact such as current "
            "production performance.\n\n"
            "Guardrail expressions look like: rmse<=baseline*1.10, bias_pct within -8..8, "
            "runtime_s<=600.\n\n"
            f"Current task.yaml:\n{self.config.config_path.read_text()}\n"
            f"Training data profile:\n{json.dumps(data_profile(self.config))}\n\n"
            f"Agent task contract (TASK.md):\n{contract}\n\n"
            f"Your skill library:\n{skill_text}"
        )

    def execute(self, name: str, arguments: dict) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the director, not the terminal
            return {"error": str(exc)}

    def _tool_finalize_brief(
        self,
        goal: str,
        context: str,
        guardrails: list[str],
        idea_hints: list[str],
        metric_name: str,
    ) -> dict:
        brief = ResearchBrief(
            goal=goal,
            context=context,
            guardrails=guardrails,
            idea_hints=idea_hints,
            metric_name=metric_name,
        )
        apply_brief(self.config, brief)
        self._reload()
        body = (
            f"[bold]Goal[/bold]\n{goal}\n\n[bold]Business context[/bold]\n{context}\n\n"
            f"[bold]Metric[/bold]\n{metric_name}\n\n[bold]Guardrails[/bold]\n"
            + "\n".join(f"- {item}" for item in guardrails)
            + "\n\n[bold]Research directions[/bold]\n"
            + "\n".join(f"- {item}" for item in idea_hints)
        )
        self.console.print(Panel(body, title="Research brief (saved to task.yaml)"))
        return {"ok": True, "saved_to": str(self.config_path)}

    def _tool_define_custom_metric(self, description: str) -> dict:
        interpreter = MetricInterpreter(
            self.config.director.model, self.config.director.temperature
        )
        spec, validation = asyncio.run(
            interpreter.interpret(description, eval_columns(self.config))
        )
        self.pending_spec = spec
        print_spec(self.console, spec, validation)
        return {
            "name": spec.name,
            "direction": spec.direction,
            "understanding": spec.understanding,
            "checks": [f"{check.name}: {check.detail}" for check in validation.checks],
        }

    def _tool_adopt_custom_metric(self) -> dict:
        if self.pending_spec is None:
            return {"error": "no custom metric has been defined yet"}
        saved = adopt_spec(self.config, self.pending_spec)
        self._reload()
        return {"ok": True, "saved_to": str(saved), "metric": self.pending_spec.name}

    def _tool_replace_baseline(self, path: str) -> dict:
        destination = replace_baseline(self.config, Path(path))
        return {"ok": True, "installed_at": str(destination)}

    def _tool_evaluate_baseline(self) -> dict:
        result = asyncio.run(validate_baseline(self.config))
        if result.error:
            return {"error": result.error}
        self.console.print(
            Panel(
                f"[bold]Validation[/bold]  {json.dumps(result.metrics)}\n"
                f"[bold]Hidden holdout[/bold]  {json.dumps(result.holdout_metrics)}",
                title="Protected baseline evaluation",
            )
        )
        return {"validation": result.metrics, "holdout": result.holdout_metrics}

    def _tool_record_context(self, text: str) -> dict:
        append_context(self.config, text)
        self._reload()
        return {"ok": True}

    def _tool_start_research(self, ideas: list[dict]) -> dict:
        if not ideas:
            return {"error": "provide at least one experiment idea"}
        known = {skill.name for skill in self.skills}
        parsed = []
        for item in ideas:
            idea = Idea.model_validate(item)
            idea.skills_used = [name for name in idea.skills_used if name in known]
            parsed.append(idea)
        self.final_ideas = parsed[: self.experiments]
        table = Table(title="Approved round 1")
        for column in ("Experiment", "Hypothesis", "Skills"):
            table.add_column(column)
        for idea in self.final_ideas:
            table.add_row(idea.title, idea.hypothesis, ", ".join(idea.skills_used) or "-")
        self.console.print(table)
        return {"ok": True, "experiments": len(self.final_ideas)}

    def run(self, first_message: str) -> list[Idea]:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": first_message},
        ]
        tool_rounds = 0
        while True:
            renderer = TurnRenderer(self.console)
            message = self.chat.turn(messages, renderer, tools=TOOLS)
            renderer.finish()
            messages.append(message)
            if message.get("tool_calls"):
                tool_rounds += 1
                if tool_rounds > MAX_TOOL_ROUNDS:
                    raise RuntimeError("the director looped on tools without finishing setup")
                for call in message["tool_calls"]:
                    name = call["function"]["name"]
                    self.console.print(f"[dim]-> {_TOOL_LABELS.get(name, name)}[/dim]")
                    try:
                        arguments = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        result: dict = {"error": "tool arguments were not valid JSON"}
                    else:
                        result = self.execute(name, arguments)
                    if "error" in result:
                        self.console.print(f"[yellow]! {result['error']}[/yellow]")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result),
                        }
                    )
                if self.final_ideas is not None:
                    return self.final_ideas
                continue
            tool_rounds = 0
            messages.append({"role": "user", "content": input_box(self.console)})
