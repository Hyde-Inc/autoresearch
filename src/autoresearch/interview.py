"""Conversational research setup: one streamed chat that configures and launches a run.

The Research Director talks freely (thinking tokens included) and acts through
tools. Pointed at a project repo, it explores the codebase and data itself -
reading files, profiling candidate datasets, running EDA - instead of asking
the user where things live. It then builds the protected task workspace,
writes the baseline (wrapping the project's existing model, or bootstrapping a
first model from EDA and its skills when there is none), evaluates it, agrees
the brief, and launches the run. Invalid input comes back as a tool error the
director explains in conversation instead of crashing the CLI.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import eda
from .chat import StreamingChat, TurnRenderer, input_box
from .config import DEFAULT_MODEL, TaskConfig, load_config
from .discover import read_repo_file, render_inventory, repo_inventory
from .editor import open_in_editor
from .metrics import MetricInterpreter, MetricSpec, MetricValidation, adopt_spec, eval_columns
from .models import Idea
from .orchestrator import validate_baseline
from .plans import (
    PlanError,
    mark_executed,
    new_session_dir,
    parse_plan,
    plan_path,
    plans_dir_for_task,
    refresh_session_readme,
    render_plan,
)
from .prepare import prepare_workspace, write_baseline
from .skills import ResearchSkill, load_skills
from .slash import (
    COMMANDS,
    MAX_PARALLEL,
    MIN_PARALLEL,
    SessionSettings,
    handle_slash,
    parse_reply,
)
from .survey import cached_survey, survey_repo

MAX_TOOL_ROUNDS = 16


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
    current = raw.get("metric") or {}
    # Never clobber an already-adopted custom metric (it carries a definition path).
    if not current.get("definition") and brief.metric_name and not brief.metric_description:
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
        "read_file",
        "Read a file from the project (README, model code, configs). Use this to learn the "
        "project yourself instead of asking the user.",
        {"path": {"type": "string", "description": "Path relative to the project root"}},
        ["path"],
    ),
    _tool(
        "explore_data",
        "Analyze a data file in the project. 'profile' describes rows/columns/dtypes and "
        "needs no column names; 'seasonality' needs date_column and target_column; "
        "'intermittency' needs id_column, date_column, and target_column; 'drivers' needs "
        "target_column.",
        {
            "path": {"type": "string"},
            "analysis": {
                "type": "string",
                "enum": ["profile", "seasonality", "intermittency", "drivers"],
            },
            "id_column": {"type": "string"},
            "date_column": {"type": "string"},
            "target_column": {"type": "string"},
        },
        ["path", "analysis"],
    ),
    _tool(
        "prepare_workspace",
        "Build the protected task from the project: split the chosen data chronologically "
        "into train/validation/holdout, copy the project code into a sealed seed workspace, "
        "and generate the task config. Re-preparing resets any baseline already written.",
        {
            "train_data": {"type": "string", "description": "Data file path in the project"},
            "id_column": {"type": "string"},
            "date_column": {"type": "string"},
            "target_column": {"type": "string"},
            "validation_days": {"type": "integer", "minimum": 1},
            "holdout_days": {"type": "integer", "minimum": 1},
            "name": {"type": "string", "description": "Optional task name"},
        },
        [
            "train_data",
            "id_column",
            "date_column",
            "target_column",
            "validation_days",
            "holdout_days",
        ],
    ),
    _tool(
        "write_baseline",
        "Write solution/train.py in the seed workspace: the incumbent baseline every "
        "experiment must beat. Provide complete Python source following the runtime "
        "contract. Wrap the project's existing model when there is one; otherwise write "
        "the initial model you chose from the EDA and your skills.",
        {"code": {"type": "string", "description": "Full Python source for train.py"}},
        ["code"],
    ),
    _tool(
        "finalize_brief",
        "Save the agreed research brief to the task config. Call only after the user has "
        "confirmed the brief in conversation.",
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
        "hidden holdout metrics. If it errors, fix the code with write_baseline and retry.",
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
        "write_plan",
        "Write the round-1 research plan to an editable markdown file in the project's "
        "research/ folder (it opens in the user's editor). The user reviews and edits that "
        "file, then replies 'execute' to launch; calling this again rewrites the same file "
        "with your revision.",
        {"ideas": _IDEA_SCHEMA},
        ["ideas"],
    ),
]

_TOOL_LABELS = {
    "read_file": "reading project files",
    "explore_data": "analyzing the data",
    "prepare_workspace": "building the protected task workspace",
    "write_baseline": "writing the baseline model",
    "finalize_brief": "saving the research brief",
    "define_custom_metric": "writing and verifying the custom metric",
    "adopt_custom_metric": "adopting the custom metric as the grader",
    "replace_baseline": "installing your baseline script",
    "evaluate_baseline": "running the protected baseline evaluation",
    "record_context": "recording that in the task context",
    "write_plan": "writing the research plan file",
}

_RUNTIME_CONTRACT = """\
Runtime contract for solution/train.py (the baseline and every experiment):
- The evaluator runs `uv run python solution/train.py` inside the seed workspace, so any
  import must be satisfied by the workspace pyproject.toml dependencies.
- Environment variables: AUTORESEARCH_TRAIN_DATA (parquet training history),
  AUTORESEARCH_REQUEST (parquet with exactly the id/date rows to forecast), and
  AUTORESEARCH_OUTPUT (destination path).
- Write a parquet file to AUTORESEARCH_OUTPUT with exactly the id column, the date column,
  and 'forecast'. Every requested row must appear exactly once; forecasts must be finite
  and non-negative.
- Training data columns match the prepared train split; the request contains only future
  dates the model has never seen."""


class SetupSession:
    """Drives the streamed setup conversation and executes the director's tools."""

    def __init__(
        self,
        client,
        console: Console,
        settings: SessionSettings | None = None,
        config_path: Path | None = None,
        repo: Path | None = None,
    ):
        if config_path is None and repo is None:
            raise ValueError("provide a task config or a project repo")
        self.console = console
        self.settings = settings or SessionSettings()
        self.config_path = config_path
        self.config = load_config(config_path) if config_path else None
        self.repo = (repo or self.config.root).resolve()
        if self.config is not None:
            model = self.config.director.model
            temperature = self.config.director.temperature
        else:
            model = DEFAULT_MODEL
            temperature = 0.35
        self.director_model = model
        self.chat = StreamingChat(client, model.removeprefix("openrouter/"), temperature)
        self.skills: list[ResearchSkill] = load_skills(self.config)
        self.pending_spec: MetricSpec | None = None
        self.final_ideas: list[Idea] | None = None
        self.survey: str | None = cached_survey(self.repo)
        self.session_dir: Path | None = None
        self.plan_path: Path | None = None
        self._survey_thread: threading.Thread | None = None
        self._survey_outcome: dict = {}

    def _reload(self) -> None:
        assert self.config_path is not None
        self.config = load_config(self.config_path)

    def _require_config(self) -> TaskConfig:
        if self.config is None:
            raise ValueError("no task workspace exists yet; call prepare_workspace first")
        return self.config

    def system_prompt(self) -> str:
        skill_text = "\n\n".join(
            f"### {skill.name}\nWhen to use: {skill.description}\n{skill.body}"
            for skill in self.skills
        )
        style = (
            "You are the Research Director of an autonomous demand forecasting lab, setting up "
            "a research run together with the user in a terminal chat.\n\n"
            "Conversation style: short messages, plain language, one question at a time. Never "
            "offer yes/no menus or numbered options; just talk. Never ask the user for anything "
            "you can discover yourself with tools - explore first, then state what you found as "
            "facts the user can correct. If the user gives an unclear or invalid answer, say "
            "what went wrong and ask again; never stop the session.\n\n"
            "The user edits run settings with slash commands (/goal, /baseline, /n_agents, "
            "/metric, /guardrail, /rounds, /timeout). Changes arrive as [settings updated] "
            "notes in the conversation. Treat them as final decisions: do not re-confirm them, "
            "and never ask the user to approve something they already set.\n\n"
            f"Current settings:\n{self.settings.describe()}\n\n"
        )
        if self.config is None:
            findings = f"Project inventory:\n{render_inventory(repo_inventory(self.repo))}"
            if self.survey:
                findings = (
                    "Repository survey (a coding agent explored the project and wrote this "
                    f"report):\n{self.survey}\n\n{findings}"
                )
            elif self.survey_pending:
                findings = (
                    "A coding agent is surveying the repository in the background; its "
                    "report will appear in this context once ready. Do not wait for it or "
                    "mention it - work from the inventory below and your read/EDA tools.\n\n"
                    f"{findings}"
                )
            flow = (
                "No task workspace exists yet. Three things must be settled before research "
                "can start: the goal, the baseline model, and the evaluation metric. "
                "Everything else you decide yourself.\n\n"
                "Your flow:\n"
                "1. Start from the repository findings below; verify anything load-bearing "
                "or surprising with read_file and profile the data with explore_data. "
                "Identify the training data, its id/date/target columns, and what models "
                "already exist.\n"
                "2. If the goal or baseline is still unclear after exploring, ask about that "
                "one thing, sharing what you found in the project while you do. If a "
                "settings note already pins them, they are decided.\n"
                "3. Settle the metric BEFORE locking in. The standard metrics (wmape, mape, "
                "rmse, bias_pct) need no confirmation. But if the user asked for a custom "
                "metric (see the settings note) or the goal implies an asymmetric or "
                "weighted objective - for example penalizing under-forecasting more than "
                "over-forecasting - call define_custom_metric with a faithful description, "
                "show the plain-English restatement, the hand-worked example, and the "
                "verification checks, and get an explicit yes. Re-run define_custom_metric if "
                "they want changes. Do not move on until a requested custom metric is "
                "confirmed by the user.\n"
                "4. The moment goal, baseline, and metric are settled, LOCK IN and run the "
                "whole launch sequence without asking for permission at any step:\n"
                "   a. Call prepare_workspace, stating your setup choices (data file, "
                "columns, validation and holdout horizon matched to the business forecast "
                "horizon) as brief facts.\n"
                "   b. If the user confirmed a custom metric, call adopt_custom_metric now so "
                "the baseline and every experiment are scored with it.\n"
                "   c. Call write_baseline. Port the pinned baseline script faithfully so "
                "the research has to beat the user's current approach. If no baseline is "
                "pinned and none exists in the repo, run the EDA you need (seasonality, "
                "intermittency, drivers) and choose a simple, robust first model from your "
                "skills; tell the user why.\n"
                "   d. Call evaluate_baseline and interpret the numbers in one or two "
                "sentences. If evaluation errors, fix the code with write_baseline and "
                "retry.\n"
                "   e. Call finalize_brief using the settings above (goal, metric, "
                "guardrails) plus the business context you learned.\n"
                f"   f. Propose exactly {self.settings.n_agents} round-1 experiment(s), "
                "each a single testable change, citing which skills informed it, and call "
                "write_plan with the ideas (fields: title, hypothesis, instructions, "
                "category, skills_used). This saves the plan as an editable markdown file. "
                "After calling it, tell the user in one short message where the plan file "
                "is and that they can edit it freely, then reply 'execute' to launch, give "
                "feedback to revise, or 'stop'. Do not launch anything yourself - the "
                "launch happens when they say execute.\n\n"
                "If the user gives feedback on the plan, revise the ideas and call "
                "write_plan again; it rewrites the same file.\n"
                "Never call write_plan until a requested custom metric has been adopted.\n\n"
                "After round 1 you will analyze results and write the next round's plan, "
                "which the user reviews the same way before it runs.\n\n"
                f"{findings}\n\n"
            )
        else:
            seed = self.config.resolve(self.config.workspace.seed)
            baseline = seed / "solution" / "train.py"
            contract_path = seed / "TASK.md"
            contract = (
                contract_path.read_text() if contract_path.exists() else self.config.description
            )
            baseline_state = (
                f"The seeded baseline is {baseline}."
                if baseline.exists()
                else "No baseline exists yet; write one with write_baseline before evaluating."
            )
            flow = (
                "A task workspace is already prepared. Your flow:\n"
                "1. Understand what the research should improve. Fill gaps with sensible "
                "defaults instead of interrogating the user. Present the brief (goal, "
                "business context, metric, guardrails, research directions) in plain text, "
                "and once the user agrees, call finalize_brief.\n"
                "2. Standard metrics are wmape, mape, rmse, bias_pct. Only if the user needs "
                "custom grading logic, call define_custom_metric, walk them through the "
                "verified result, and call adopt_custom_metric after they agree.\n"
                f"3. {baseline_state} Only if the user offers their own script, ask for its "
                "path and call replace_baseline. Then call evaluate_baseline and interpret "
                "the numbers for the user in one or two sentences.\n"
                f"4. Propose a round-1 plan of exactly {self.settings.n_agents} "
                "experiment(s), each a single testable change, citing which skills informed "
                "it, and call write_plan with the ideas (fields: title, hypothesis, "
                "instructions, category, skills_used). Tell the user where the plan file "
                "is; they edit it and reply 'execute' to launch, give feedback to revise "
                "(call write_plan again), or 'stop'.\n\n"
                f"Current task config:\n{self.config.config_path.read_text()}\n"
                f"Training data profile:\n{json.dumps(data_profile(self.config))}\n\n"
                f"Agent task contract (TASK.md):\n{contract}\n\n"
            )
            if self.survey:
                flow += (
                    "Repository survey (a coding agent explored the original project):\n"
                    f"{self.survey}\n\n"
                )
        return (
            style
            + flow
            + "Use record_context whenever the user shares a useful business fact such as "
            "current production performance.\n\n"
            "Guardrail expressions look like: rmse<=baseline*1.10, bias_pct within -8..8, "
            "runtime_s<=600.\n\n" + _RUNTIME_CONTRACT + f"\n\nYour skill library:\n{skill_text}"
        )

    def execute(self, name: str, arguments: dict) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        try:
            return handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the director, not the terminal
            return {"error": str(exc)}

    def _tool_read_file(self, path: str) -> dict:
        return {"path": path, "content": read_repo_file(self.repo, path)}

    def _tool_explore_data(
        self,
        path: str,
        analysis: str,
        id_column: str | None = None,
        date_column: str | None = None,
        target_column: str | None = None,
    ) -> dict:
        target = (self.repo / path).resolve()
        if self.repo not in target.parents:
            return {"error": f"path escapes the project: {path}"}
        frame = eda.load_table(target)
        if analysis == "profile":
            return eda.profile(frame)
        if analysis == "seasonality":
            if not (date_column and target_column):
                return {"error": "seasonality needs date_column and target_column"}
            return eda.seasonality(frame, date_column, target_column)
        if analysis == "intermittency":
            if not (id_column and date_column and target_column):
                return {"error": "intermittency needs id_column, date_column, target_column"}
            return eda.intermittency(frame, id_column, date_column, target_column)
        if analysis == "drivers":
            if not target_column:
                return {"error": "drivers needs target_column"}
            return eda.drivers(
                frame, target_column, exclude=[c for c in (id_column, date_column) if c]
            )
        return {"error": f"unknown analysis: {analysis}"}

    def _tool_prepare_workspace(
        self,
        train_data: str,
        id_column: str,
        date_column: str,
        target_column: str,
        validation_days: int,
        holdout_days: int,
        name: str | None = None,
    ) -> dict:
        prepared = prepare_workspace(
            self.repo,
            train_data=train_data,
            id_column=id_column,
            date_column=date_column,
            target_column=target_column,
            validation_days=int(validation_days),
            holdout_days=int(holdout_days),
            name=name,
            overrides=self.settings.overrides(),
        )
        self.config_path = prepared.config_path
        self._reload()
        self.skills = load_skills(self.config)
        self.console.print(
            Panel(
                f"[bold]Train[/bold] {prepared.train_rows:,} rows through {prepared.train_end}\n"
                f"[bold]Validation[/bold] {prepared.validation_rows:,} rows through "
                f"{prepared.validation_end}\n"
                f"[bold]Hidden holdout[/bold] {prepared.holdout_rows:,} rows through "
                f"{prepared.holdout_end}\n"
                f"[bold]Items[/bold] {prepared.items}",
                title=f"Protected task workspace ({prepared.task_dir})",
            )
        )
        return {
            "ok": True,
            "task_dir": str(prepared.task_dir),
            "train_rows": prepared.train_rows,
            "validation_rows": prepared.validation_rows,
            "holdout_rows": prepared.holdout_rows,
            "items": prepared.items,
            "next": "write the baseline with write_baseline",
        }

    def _tool_write_baseline(self, code: str) -> dict:
        destination = write_baseline(self._require_config(), code)
        lines = len(code.splitlines())
        self.console.print(f"[dim]Baseline written: {destination} ({lines} lines)[/dim]")
        return {"ok": True, "written_to": str(destination), "lines": lines}

    def _tool_finalize_brief(
        self,
        goal: str,
        context: str,
        guardrails: list[str],
        idea_hints: list[str],
        metric_name: str,
    ) -> dict:
        config = self._require_config()
        brief = ResearchBrief(
            goal=goal,
            context=context,
            guardrails=guardrails,
            idea_hints=idea_hints,
            metric_name=metric_name,
        )
        apply_brief(config, brief)
        self._reload()
        body = (
            f"[bold]Goal[/bold]\n{goal}\n\n[bold]Business context[/bold]\n{context}\n\n"
            f"[bold]Metric[/bold]\n{metric_name}\n\n[bold]Guardrails[/bold]\n"
            + "\n".join(f"- {item}" for item in guardrails)
            + "\n\n[bold]Research directions[/bold]\n"
            + "\n".join(f"- {item}" for item in idea_hints)
        )
        self.console.print(Panel(body, title="Research brief (saved)"))
        return {"ok": True, "saved_to": str(self.config_path)}

    def _tool_define_custom_metric(self, description: str) -> dict:
        # Runnable before the workspace exists so the metric can be confirmed up front.
        # The worked-example verification does not need the real data; columns are advisory.
        if self.config is not None:
            model = self.config.director.model
            temperature = self.config.director.temperature
            columns = eval_columns(self.config)
        else:
            model = DEFAULT_MODEL
            temperature = 0.35
            columns = ["actual", "forecast"]
        interpreter = MetricInterpreter(model, temperature)
        spec, validation = asyncio.run(interpreter.interpret(description, columns))
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
        saved = adopt_spec(self._require_config(), self.pending_spec)
        self._reload()
        return {"ok": True, "saved_to": str(saved), "metric": self.pending_spec.name}

    def _tool_replace_baseline(self, path: str) -> dict:
        destination = replace_baseline(self._require_config(), Path(path))
        return {"ok": True, "installed_at": str(destination)}

    def _tool_evaluate_baseline(self) -> dict:
        result = asyncio.run(validate_baseline(self._require_config()))
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
        append_context(self._require_config(), text)
        self._reload()
        return {"ok": True}

    def _tool_write_plan(self, ideas: list[dict]) -> dict:
        config = self._require_config()
        if self.settings.metric_description and config.metric.definition is None:
            return {
                "error": "a custom metric was requested but has not been adopted yet; call "
                "define_custom_metric, get the user's confirmation, then adopt_custom_metric "
                "before writing the plan",
            }
        if not ideas:
            return {"error": "provide at least one experiment idea"}
        known = {skill.name for skill in self.skills}
        parsed = []
        for item in ideas:
            idea = Idea.model_validate(item)
            idea.skills_used = [name for name in idea.skills_used if name in known]
            parsed.append(idea)
        parsed = parsed[: self.settings.n_agents]
        if self.plan_path is None:
            self.session_dir = new_session_dir(
                plans_dir_for_task(config.root), self.settings.goal or config.goal
            )
            self.plan_path = plan_path(self.session_dir, round_number=1)
        self.plan_path.write_text(
            render_plan(
                round_number=1,
                ideas=parsed,
                goal=self.settings.goal or config.goal,
                metric=config.metric.name,
                baseline=self.settings.baseline_path or "",
                n_agents=len(parsed),
                guardrails=config.guardrails,
                timeout_s=self.settings.timeout_s,
            )
        )
        refresh_session_readme(self.session_dir)
        opened = open_in_editor(self.plan_path)
        table = Table(title=f"Proposed round 1 ({len(parsed)} researches in parallel)")
        for column in ("#", "Experiment", "Hypothesis", "Skills"):
            table.add_column(column)
        for index, idea in enumerate(parsed, 1):
            table.add_row(
                str(index), idea.title, idea.hypothesis, ", ".join(idea.skills_used) or "-"
            )
        self.console.print(table)
        opened_note = "Opened in your editor.\n" if opened else ""
        self.console.print(
            Panel(
                f"[bold]{self.plan_path}[/bold]\n"
                f"{opened_note}"
                "Edit the file freely - reword, delete, or add experiments. The edited file "
                "is exactly what runs.\n"
                "Reply [green]execute[/green] to launch, [red]stop[/red] to end, or give "
                "feedback to revise the plan.",
                title="Research plan written",
            )
        )
        return {
            "ok": True,
            "plan_path": str(self.plan_path),
            "experiments": len(parsed),
            "next": "tell the user where the plan file is and wait for execute/feedback/stop",
        }

    def _apply_plan_overrides(self, overrides: dict) -> None:
        """Fold the human's frontmatter edits back into settings and the task config."""
        if overrides.get("goal"):
            self.settings.goal = str(overrides["goal"])
        if overrides.get("n_agents"):
            self.settings.n_agents = max(
                MIN_PARALLEL, min(int(overrides["n_agents"]), MAX_PARALLEL)
            )
        if overrides.get("guardrails"):
            self.settings.guardrails = [str(item) for item in overrides["guardrails"]]
        if overrides.get("timeout_s"):
            self.settings.timeout_s = max(60, int(overrides["timeout_s"]))
        config = self._require_config()
        raw = yaml.safe_load(config.config_path.read_text())
        if self.settings.goal:
            raw["goal"] = self.settings.goal
        raw["guardrails"] = self.settings.guardrails
        agents = raw.get("agents") or {}
        agents["count"] = self.settings.n_agents
        agents["timeout_s"] = self.settings.timeout_s
        raw["agents"] = agents
        metric = overrides.get("metric")
        current = raw.get("metric") or {}
        if metric in {"wmape", "mape", "rmse", "bias_pct"} and not current.get("definition"):
            raw["metric"] = {"name": metric, "direction": "min"}
        config.config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        self._reload()

    @property
    def survey_pending(self) -> bool:
        return self._survey_thread is not None

    def start_survey(self) -> None:
        """Send a coding agent to survey the repo in the background, Cursor-subagent
        style: the chat opens immediately and the findings merge in when ready."""
        if self.config is not None or self.survey or self._survey_thread:
            return

        def work() -> None:
            try:
                self._survey_outcome["text"] = survey_repo(self.repo, self.director_model)
            except Exception as exc:  # noqa: BLE001 - survey is best-effort
                self._survey_outcome["error"] = str(exc)

        self._survey_thread = threading.Thread(target=work, name="repo-survey", daemon=True)
        self._survey_thread.start()
        self.console.print(
            "[dim]A coding agent is surveying the repository in the background - "
            "start typing, its findings will be folded in when ready.[/dim]"
        )

    def poll_survey(self) -> bool:
        """Collect a finished background survey. Returns True when new findings landed."""
        if self._survey_thread is None or self._survey_thread.is_alive():
            return False
        self._survey_thread = None
        self.survey = self._survey_outcome.get("text")
        if self.survey:
            self.console.print(
                "[dim]Repo survey ready - report added to the director's context "
                "(saved to .autoresearch/survey.md).[/dim]"
            )
            return True
        error = self._survey_outcome.get("error")
        reason = f"survey failed ({error})" if error else "the agent produced no report"
        self.console.print(
            f"[dim]Repo survey skipped: {reason}; the director continues with the "
            "static inventory and its own tools.[/dim]"
        )
        return False

    def next_user_message(self) -> str:
        """Read input, applying slash commands locally until a chat message arrives."""
        notes: list[str] = []
        while True:
            text = input_box(self.console, completions=COMMANDS)
            result = handle_slash(text, self.settings, self.console)
            if not result.handled:
                if notes:
                    return "[settings updated]\n" + "\n".join(notes) + "\n\n" + text
                return text
            if result.note:
                notes.append(result.note)

    def _gate_user_message(self) -> str | None:
        """Next message for the director, or None when the plan gate resolved the session.

        Once a plan file exists, 'execute' parses the (possibly hand-edited) file and
        ends setup with the final ideas; 'stop' ends with none; anything else is
        feedback passed through to the director.
        """
        while True:
            text = self.next_user_message()
            if self.plan_path is None:
                return text
            verdict = parse_reply(text)
            if verdict == "revise":
                return text
            if verdict == "stop":
                self.final_ideas = []
                return None
            try:
                parsed = parse_plan(self.plan_path)
            except PlanError as exc:
                self.console.print(
                    f"[yellow]! {exc}[/yellow]\n"
                    "Fix the plan file and type execute again, or give feedback to revise it."
                )
                continue
            self._apply_plan_overrides(parsed.overrides)
            mark_executed(self.plan_path)
            self.final_ideas = parsed.ideas
            return None

    def run(self) -> list[Idea]:
        self.start_survey()
        first = self.next_user_message()
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": first},
        ]
        tool_rounds = 0
        while True:
            if self.poll_survey():
                messages[0] = {"role": "system", "content": self.system_prompt()}
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
                continue
            tool_rounds = 0
            reply = self._gate_user_message()
            if reply is None:
                return self.final_ideas or []
            messages.append({"role": "user", "content": reply})
