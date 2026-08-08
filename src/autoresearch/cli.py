from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_config
from .director import ResearchDirector
from .ingest import ingest_csv, preview_csv
from .interview import (
    ResearchBrief,
    ResearchInterview,
    append_context,
    apply_brief,
    replace_baseline,
)
from .metrics import MetricInterpreter, MetricSpec, MetricValidation, adopt_spec, eval_columns
from .orchestrator import run_research, validate_baseline
from .report import build_report
from .skills import load_skills
from .store import RunStore, latest_run

app = typer.Typer(no_args_is_help=True, help="Parallel autonomous ML experimentation.")
console = Console()
ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Task YAML")
]


def _store_from(run_dir: Path | None, config: Path | None = None) -> RunStore:
    if run_dir:
        return RunStore(run_dir)
    if config:
        cfg = load_config(config)
        found = latest_run(cfg.resolve(cfg.workspace.runs), cfg.name)
    else:
        found = latest_run(Path("runs"))
    if found is None:
        raise typer.BadParameter("no run found; pass --run-dir")
    return RunStore(found)


@app.command()
def run(
    config: ConfigOption,
    goal: Annotated[str | None, typer.Option(help="Override the configured goal")] = None,
    guardrail: Annotated[
        list[str] | None, typer.Option("--guardrail", help="Additional guardrail expression")
    ] = None,
    parallel: Annotated[int | None, typer.Option(min=1)] = None,
    max_experiments: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Start a new autonomous research run."""
    load_dotenv()
    cfg = load_config(config)
    store = asyncio.run(
        run_research(
            cfg,
            goal=goal,
            guardrails=guardrail,
            parallel=parallel,
            max_experiments=max_experiments,
        )
    )
    console.print(f"Run complete: [bold]{store.run_dir}[/bold]")


@app.command()
def resume(
    config: ConfigOption,
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    parallel: Annotated[int | None, typer.Option(min=1)] = None,
    max_experiments: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Resume the latest or selected run from its saved round state."""
    load_dotenv()
    cfg = load_config(config)
    store = _store_from(run_dir, config)
    completed = asyncio.run(
        run_research(
            cfg,
            parallel=parallel,
            max_experiments=max_experiments,
            resume_store=store,
        )
    )
    console.print(f"Run complete: [bold]{completed.run_dir}[/bold]")


@app.command()
def ingest(
    csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Sales CSV")],
    name: Annotated[str, typer.Option("--name", "-n", help="Task name")],
    tasks_root: Annotated[Path, typer.Option(help="Task output directory")] = Path("tasks"),
    validation_days: Annotated[int | None, typer.Option(min=1)] = None,
    holdout_days: Annotated[int | None, typer.Option(min=1)] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing task")] = False,
) -> None:
    """Turn a sales CSV into a protected forecasting task."""
    preview = preview_csv(csv)
    for warning in preview.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    if not preview.ok:
        for error in preview.errors:
            console.print(f"[red]x {error}[/red]")
        raise typer.Exit(1)
    console.print(
        f"Found [bold]{preview.rows}[/bold] rows, [bold]{preview.skus}[/bold] items, "
        f"{preview.distinct_dates} dates ({preview.date_min} to {preview.date_max})"
    )
    result = ingest_csv(
        csv,
        name=name,
        tasks_root=tasks_root,
        validation_days=validation_days,
        holdout_days=holdout_days,
        overwrite=overwrite,
    )
    console.print(f"[green]Created task[/green] {result.config_path}")


def _print_spec(spec: MetricSpec, validation: MetricValidation) -> None:
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


@app.command()
def metric(
    config: ConfigOption,
    description: Annotated[
        str, typer.Argument(help="Plain-English evaluation metric")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Create and verify a custom metric from plain English."""
    load_dotenv()
    cfg = load_config(config)
    interpreter = MetricInterpreter(cfg.director.model, cfg.director.temperature)
    spec, validation = asyncio.run(interpreter.interpret(description, eval_columns(cfg)))
    _print_spec(spec, validation)
    if not yes and not typer.confirm("Use this metric for every experiment?"):
        console.print("Metric discarded.")
        return
    saved = adopt_spec(cfg, spec)
    console.print(f"[green]Metric adopted.[/green] Saved to {saved}")


@app.command("skills")
def list_skills(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """List the forecasting knowledge available to the Research Director."""
    cfg = load_config(config) if config else None
    table = Table(title="Research Director skills")
    table.add_column("Skill")
    table.add_column("When it is used")
    table.add_column("Source")
    for skill in load_skills(cfg):
        table.add_row(skill.name, skill.description, skill.source.name)
    console.print(table)


def _print_brief(brief: ResearchBrief) -> None:
    lines = [
        f"[bold]Goal[/bold]\n{brief.goal}",
        f"[bold]Business context[/bold]\n{brief.context}",
        f"[bold]Metric[/bold]\n{brief.metric_name}"
        + (f": {brief.metric_description}" if brief.metric_description else ""),
        "[bold]Guardrails[/bold]\n" + "\n".join(f"- {item}" for item in brief.guardrails),
        "[bold]Research directions[/bold]\n"
        + "\n".join(f"- {item}" for item in brief.idea_hints),
    ]
    console.print(Panel("\n\n".join(lines), title="Research brief"))


def _print_plan(director: ResearchDirector, ideas: list) -> None:
    if director.last_skill_selection.selected:
        console.print("\n[bold]Director consulted skills[/bold]")
        for selected in director.last_skill_selection.selected:
            console.print(f"  [cyan]{selected.name}[/cyan]: {selected.reason}")
    table = Table(title="Proposed round 1")
    for column in ("Experiment", "Hypothesis", "Skills"):
        table.add_column(column)
    for idea in ideas:
        table.add_row(idea.title, idea.hypothesis, ", ".join(idea.skills_used) or "-")
    console.print(table)


@app.command()
def start(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False)
    ] = None,
    csv: Annotated[
        Path | None, typer.Option("--csv", exists=True, dir_okay=False)
    ] = None,
    name: Annotated[str | None, typer.Option("--name", "-n")] = None,
    tasks_root: Annotated[Path, typer.Option(help="Task output directory")] = Path("tasks"),
    parallel: Annotated[int | None, typer.Option(min=1)] = None,
    max_experiments: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Design the research assignment together, then start the agents."""
    load_dotenv()
    console.print(Panel("Define the goal, metric, baseline, and first research plan together."))
    if config and csv:
        raise typer.BadParameter("pass either --config or --csv, not both")
    if not config and not csv:
        source = Path(typer.prompt("Path to task.yaml or sales CSV")).expanduser()
        if source.suffix.lower() == ".csv":
            csv = source
        else:
            config = source
    if csv:
        preview = preview_csv(csv)
        if not preview.ok:
            raise typer.BadParameter("; ".join(preview.errors))
        console.print(
            f"Found {preview.rows} rows, {preview.skus} items, "
            f"{preview.distinct_dates} dates."
        )
        if not typer.confirm("Create chronological train, validation, and hidden holdout splits?"):
            raise typer.Abort()
        task_name = name or typer.prompt("Task name")
        result = ingest_csv(csv, name=task_name, tasks_root=tasks_root)
        config = result.config_path
    if config is None or not config.exists():
        raise typer.BadParameter(f"task config not found: {config}")

    cfg = load_config(config)
    director = ResearchDirector(cfg)
    interview = ResearchInterview(cfg, director._json_completion)
    conversation = [
        {
            "role": "user",
            "content": typer.prompt("What should this research improve?"),
        }
    ]
    brief: ResearchBrief | None = None
    while brief is None:
        turn = asyncio.run(interview.next_turn(conversation))
        console.print(f"\n[bold cyan]Research Director[/bold cyan]\n{turn.message}")
        if turn.done:
            brief = turn.brief
            break
        answer = typer.prompt("You (type /done to finish)")
        conversation.append({"role": "assistant", "content": turn.message})
        conversation.append({"role": "user", "content": answer})
        if answer.strip().lower() == "/done":
            turn = asyncio.run(interview.next_turn(conversation, force_finish=True))
            brief = turn.brief
    assert brief is not None
    while True:
        _print_brief(brief)
        if typer.confirm("Use this research brief?"):
            break
        revision = typer.prompt("What should change?")
        conversation.append({"role": "user", "content": revision})
        turn = asyncio.run(interview.next_turn(conversation, force_finish=True))
        if turn.brief is None:
            raise RuntimeError("Research Director did not return a revised brief")
        brief = turn.brief
    apply_brief(cfg, brief)
    cfg = load_config(config)

    if brief.metric_description:
        console.print("\n[bold]Verifying the custom metric[/bold]")
        interpreter = MetricInterpreter(cfg.director.model, cfg.director.temperature)
        spec, validation = asyncio.run(
            interpreter.interpret(brief.metric_description, eval_columns(cfg))
        )
        _print_spec(spec, validation)
        if typer.confirm("Use this verified metric?"):
            adopt_spec(cfg, spec)
            cfg = load_config(config)

    baseline = cfg.resolve(cfg.workspace.seed) / "solution" / "train.py"
    console.print(f"\n[bold]Baseline model[/bold]\n{baseline}")
    if typer.confirm("Replace the seeded baseline with your own train.py?", default=False):
        replacement = Path(typer.prompt("Path to train.py"))
        baseline = replace_baseline(cfg, replacement)
        console.print(f"Using {baseline}")
    production = typer.prompt(
        "Current production performance (optional, press Enter to skip)",
        default="",
        show_default=False,
    )
    if production.strip():
        append_context(cfg, f"Current production baseline reported by the team: {production}")
        cfg = load_config(config)
    console.print("Running the protected baseline evaluation...")
    baseline_result = asyncio.run(validate_baseline(cfg))
    if baseline_result.error:
        console.print(f"[red]Baseline failed:[/red] {baseline_result.error}")
        raise typer.Exit(1)
    console.print(f"Validation: {json.dumps(baseline_result.metrics)}")
    console.print(f"Hidden holdout: {json.dumps(baseline_result.holdout_metrics)}")

    director = ResearchDirector(cfg)
    feedback = ""
    count = parallel or cfg.agents.count
    while True:
        ideas = asyncio.run(
            director.propose(
                count=count,
                round_number=1,
                attempts=[],
                notes=feedback,
            )
        )
        _print_plan(director, ideas)
        if typer.confirm("Start research with this plan?"):
            break
        feedback = typer.prompt("What should the Director change?")
    store = asyncio.run(
        run_research(
            cfg,
            parallel=parallel,
            max_experiments=max_experiments,
            initial_ideas=ideas,
        )
    )
    console.print(f"Run complete: [bold]{store.run_dir}[/bold]")


@app.command()
def validate(config: ConfigOption) -> None:
    """Run the seed solution through the protected evaluator."""
    cfg = load_config(config)
    result = asyncio.run(validate_baseline(cfg))
    if result.error:
        console.print(f"[red]Validation failed:[/red] {result.error}")
        raise typer.Exit(1)
    console.print("[green]Baseline is valid.[/green]")
    console.print(f"validation={json.dumps(result.metrics, indent=2)}")
    console.print(f"holdout={json.dumps(result.holdout_metrics, indent=2)}")
    if result.guardrail_failures:
        console.print(f"[yellow]Guardrail failures:[/yellow] {result.guardrail_failures}")


@app.command()
def leaderboard(
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    config: Annotated[Path | None, typer.Option("-c", exists=True, dir_okay=False)] = None,
) -> None:
    """Show ranked, guardrail-passing attempts."""
    store = _store_from(run_dir, config)
    state = store.load_state()
    metric = state.get("primary_metric", "wmape")
    table = Table(title=f"{state.get('task', 'Autoresearch')} leaderboard")
    for column in (
        "Rank", "ID", "Experiment", "Skills", metric.upper(), "RMSE", "Holdout", "Promoted"
    ):
        table.add_column(column)
    for rank, item in enumerate(
        store.leaderboard(metric, state.get("metric_direction", "min")), 1
    ):
        table.add_row(
            str(rank),
            item.id,
            item.idea.title,
            ", ".join(item.idea.skills_used),
            f"{item.metrics.get(metric, float('nan')):.6f}",
            f"{item.metrics.get('rmse', float('nan')):.3f}",
            f"{item.holdout_metrics.get('wmape', float('nan')):.6f}",
            "yes" if item.promoted else "",
        )
    console.print(table)


@app.command()
def show(
    attempt_id: str,
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
) -> None:
    """Show one attempt and its diff."""
    store = _store_from(run_dir)
    attempts = {item.id: item for item in store.load_attempts()}
    if attempt_id not in attempts:
        raise typer.BadParameter(f"unknown attempt: {attempt_id}")
    attempt = attempts[attempt_id]
    console.print_json(attempt.model_dump_json())
    if attempt.diff_path and Path(attempt.diff_path).exists():
        console.print(Path(attempt.diff_path).read_text())


@app.command()
def report(
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    output: Annotated[Path | None, typer.Option("-o")] = None,
) -> None:
    """Generate a markdown research report."""
    store = _store_from(run_dir)
    text = build_report(store.run_dir)
    if output:
        output.write_text(text)
        console.print(f"Wrote {output}")
    else:
        console.print(text)


@app.command()
def status(
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
) -> None:
    """Show run state and experiment counts."""
    store = _store_from(run_dir)
    console.print_json(json.dumps(store.load_state()))


@app.command()
def stop() -> None:
    """Request that a running orchestrator stop after its current round."""
    Path(".autoresearch-stop").write_text("stop\n")
    console.print("Stop requested.")


if __name__ == "__main__":
    app()
