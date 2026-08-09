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

from .chat import input_box
from .config import load_config
from .director import openrouter_client
from .ingest import ingest_csv, preview_csv
from .interview import SetupSession, print_spec
from .metrics import MetricInterpreter, adopt_spec, eval_columns
from .models import Idea
from .orchestrator import ReviewDecision, run_research, validate_baseline
from .report import build_report
from .skills import load_skills
from .slash import SessionSettings, parse_reply, print_help
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
    """Resume the latest or selected run: review the next proposed round, then continue."""
    load_dotenv()
    cfg = load_config(config)
    store = _store_from(run_dir, config)
    completed = asyncio.run(
        run_research(
            cfg,
            parallel=parallel,
            max_experiments=max_experiments,
            resume_store=store,
            review=review_round,
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
    print_spec(console, spec, validation)
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


def parse_review_reply(text: str) -> ReviewDecision:
    """Whole-message approval/stop phrases decide; anything else is revision feedback."""
    verdict = parse_reply(text)
    if verdict == "approve":
        return ReviewDecision("approve")
    if verdict == "stop":
        return ReviewDecision("stop")
    return ReviewDecision("revise", feedback=text)


def review_round(round_number: int, ideas: list[Idea], plan_path: Path) -> ReviewDecision:
    """Console review gate: show the proposed round and its editable plan file."""
    table = Table(title=f"Proposed round {round_number} ({len(ideas)} researches in parallel)")
    for column in ("#", "Experiment", "Hypothesis", "Skills"):
        table.add_column(column)
    for index, idea in enumerate(ideas, 1):
        table.add_row(
            str(index), idea.title, idea.hypothesis, ", ".join(idea.skills_used) or "-"
        )
    console.print(table)
    console.print(
        Panel(
            f"[bold]{plan_path}[/bold]\n"
            "Edit the file freely - the edited file is exactly what runs.\n"
            "Reply [green]execute[/green] to launch, [red]stop[/red] to end the run, or "
            "give feedback to revise the plan.",
            title=f"Round {round_number} plan written",
        )
    )
    return parse_review_reply(input_box(console))


@app.command()
def start(
    repo: Annotated[
        Path | None,
        typer.Argument(
            exists=True, file_okay=False, help="Project repo to research (default: cwd)"
        ),
    ] = None,
) -> None:
    """Point the Research Director at a project, lock in goal and baseline, and research."""
    load_dotenv()
    repo = (repo or Path(".")).resolve()
    settings = SessionSettings()
    session = SetupSession(openrouter_client(), console, settings=settings, repo=repo)
    console.print(f"Project: [bold]{repo}[/bold]")
    console.print(
        Panel(
            "The Research Director surveys your project itself. Set what it needs with "
            "slash commands or just say it:\n"
            "  [bold cyan]/goal[/bold cyan] reduce wmape        "
            "[bold cyan]/baseline[/bold cyan] models/arima.py\n"
            "Once goal, baseline, and metric are settled it evaluates the baseline and "
            "writes the round-1 research plan to an editable markdown file in "
            ".autoresearch/plans/. You edit it, type execute, and the round runs. Every "
            "round gets its own plan file with results appended - a lab notebook of all "
            "the research ever tried.",
            title="Research setup",
        )
    )
    print_help(console)
    console.print("\nWhat should this research improve?")
    try:
        ideas = session.run()
    except KeyboardInterrupt:
        console.print("\nSetup cancelled. Nothing is running.")
        raise typer.Exit(130) from None
    except Exception as exc:  # noqa: BLE001 - CLI should show a concise failure
        console.print(f"[bold red]Setup failed:[/bold red] {exc}")
        raise typer.Exit(1) from None
    if not ideas:
        console.print("Setup ended without launching. Nothing is running.")
        return

    assert session.config_path is not None
    try:
        store = asyncio.run(
            run_research(
                load_config(session.config_path),
                initial_ideas=ideas,
                initial_plan_path=session.plan_path,
                review=review_round,
            )
        )
    except KeyboardInterrupt:
        console.print("\nRun interrupted. Resume later with: autoresearch resume -c "
                      f"{session.config_path}")
        raise typer.Exit(130) from None
    except Exception as exc:  # noqa: BLE001 - CLI should show a concise failure
        console.print(f"[bold red]Research run failed:[/bold red] {exc}")
        raise typer.Exit(1) from None
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
