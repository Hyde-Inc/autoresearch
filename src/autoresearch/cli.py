from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .chat import input_box
from .config import load_config
from .costs import format_cost
from .director import openrouter_client
from .doctor import run_checks
from .foundry import FoundryError, parse_dataset_reference, read_dataset
from .ingest import Preview, apply_column_mapping, ingest_frame, preview_frame
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


def _load_env() -> None:
    """Load .env by searching upward from the *current directory*.

    ``load_dotenv()``'s default searches from the installed package's own
    directory, which works in a repo checkout (the .venv sits next to .env)
    but silently finds nothing when the CLI is installed globally as a tool.
    """
    load_dotenv(find_dotenv(usecwd=True))


def _print_quality_gate(preview: Preview) -> None:
    """Show a per-column data-quality summary and any cleaning warnings."""
    if preview.raw_rows:
        table = Table(title="Data quality", show_edge=False)
        table.add_column("Column")
        table.add_column("Unusable rows", justify="right")
        table.add_column("% of source", justify="right")
        for column, count in preview.null_counts.items():
            pct = (count / preview.raw_rows * 100) if preview.raw_rows else 0.0
            style = "red" if pct >= 20 else ("yellow" if count else "green")
            table.add_row(column, f"[{style}]{count}[/{style}]", f"[{style}]{pct:.1f}%[/{style}]")
        console.print(table)
        drop_style = "red" if preview.dropped_pct >= 20 else "yellow" if preview.dropped_rows else "green"
        console.print(
            f"[{drop_style}]{preview.dropped_rows} of {preview.raw_rows} rows "
            f"({preview.dropped_pct:.1f}%) dropped during cleaning[/{drop_style}]"
        )
    for warning in preview.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")


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
    max_cost: Annotated[
        float | None,
        typer.Option(min=0.0, help="Stop before a new round once model spend (USD) hits this"),
    ] = None,
) -> None:
    """Start a new autonomous research run."""
    _load_env()
    cfg = load_config(config)
    store = asyncio.run(
        run_research(
            cfg,
            goal=goal,
            guardrails=guardrail,
            parallel=parallel,
            max_experiments=max_experiments,
            max_cost=max_cost,
        )
    )
    console.print(f"Run complete: [bold]{store.run_dir}[/bold]")


@app.command()
def resume(
    config: ConfigOption,
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    parallel: Annotated[int | None, typer.Option(min=1)] = None,
    max_experiments: Annotated[int | None, typer.Option(min=1)] = None,
    max_cost: Annotated[
        float | None,
        typer.Option(min=0.0, help="Stop before a new round once model spend (USD) hits this"),
    ] = None,
) -> None:
    """Resume the latest or selected run: review the next proposed round, then continue."""
    _load_env()
    cfg = load_config(config)
    store = _store_from(run_dir, config)
    completed = asyncio.run(
        run_research(
            cfg,
            parallel=parallel,
            max_experiments=max_experiments,
            max_cost=max_cost,
            resume_store=store,
            review=review_round,
        )
    )
    console.print(f"Run complete: [bold]{completed.run_dir}[/bold]")


@app.command()
def ingest(
    source: Annotated[
        str,
        typer.Argument(
            help="Sales CSV path, or a Foundry dataset: foundry://ri.foundry.main.dataset.<uuid>"
        ),
    ],
    name: Annotated[str, typer.Option("--name", "-n", help="Task name")],
    tasks_root: Annotated[Path, typer.Option(help="Task output directory")] = Path("tasks"),
    validation_days: Annotated[int | None, typer.Option(min=1)] = None,
    holdout_days: Annotated[int | None, typer.Option(min=1)] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing task")] = False,
    branch: Annotated[
        str | None, typer.Option(help="Foundry branch to read (default: the dataset default)")
    ] = None,
    id_column: Annotated[
        str | None, typer.Option(help="Source column to use as sku_name")
    ] = None,
    date_column: Annotated[str | None, typer.Option(help="Source column to use as date")] = None,
    target_column: Annotated[
        str | None, typer.Option(help="Source column to use as sales")
    ] = None,
    price_column: Annotated[
        str | None, typer.Option(help="Source column to use as selling_price")
    ] = None,
    max_drop_pct: Annotated[
        float,
        typer.Option(min=0.0, max=100.0, help="Refuse ingest if more than this % of rows drop"),
    ] = 30.0,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the data-quality confirmation prompt")
    ] = False,
) -> None:
    """Turn a sales history (local CSV or Foundry dataset) into a protected forecasting task."""
    _load_env()
    try:
        rid = parse_dataset_reference(source)
    except FoundryError as exc:
        console.print(f"[red]x {exc}[/red]")
        raise typer.Exit(1) from None

    if rid:
        console.print(f"Reading Foundry dataset [bold]{rid}[/bold]" + (f" ({branch})" if branch else ""))
        try:
            frame = read_dataset(rid, branch=branch)
        except FoundryError as exc:
            console.print(f"[red]x {exc}[/red]")
            raise typer.Exit(1) from None
        console.print(f"Downloaded [bold]{len(frame)}[/bold] rows from Foundry")
    else:
        path = Path(source)
        if not path.is_file():
            raise typer.BadParameter(f"'{source}' is not a file or a Foundry dataset reference")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - CLI should show a concise failure
            console.print(f"[red]x could not read CSV: {exc}[/red]")
            raise typer.Exit(1) from None

    try:
        frame = apply_column_mapping(
            frame,
            id_column=id_column,
            date_column=date_column,
            target_column=target_column,
            price_column=price_column,
        )
    except ValueError as exc:
        console.print(f"[red]x {exc}[/red]")
        raise typer.Exit(1) from None

    preview = preview_frame(frame)
    _print_quality_gate(preview)
    if not preview.ok:
        for error in preview.errors:
            console.print(f"[red]x {error}[/red]")
        raise typer.Exit(1)
    console.print(
        f"Found [bold]{preview.rows}[/bold] clean rows, [bold]{preview.skus}[/bold] items, "
        f"{preview.distinct_dates} dates ({preview.date_min} to {preview.date_max})"
    )
    if preview.dropped_pct > max_drop_pct:
        console.print(
            f"[red]x {preview.dropped_pct:.1f}% of rows would be dropped, above the "
            f"--max-drop-pct {max_drop_pct:.0f}% limit. Fix the source or raise the limit.[/red]"
        )
        raise typer.Exit(1)
    if not yes and not typer.confirm("Proceed with ingestion using the cleaned data?"):
        console.print("Ingestion cancelled. No task was created.")
        raise typer.Exit(0)
    result = ingest_frame(
        frame,
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
    description: Annotated[str, typer.Argument(help="Plain-English evaluation metric")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Create and verify a custom metric from plain English."""
    _load_env()
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
        table.add_row(str(index), idea.title, idea.hypothesis, ", ".join(idea.skills_used) or "-")
    console.print(table)
    console.print(
        Panel(
            f"[bold]{plan_path}[/bold] (opened in your editor)\n"
            "Edit the file freely - the edited file is exactly what runs.\n"
            "Reply [green]execute[/green] to launch, [red]stop[/red] to end the run, or "
            "give feedback to revise the plan.\n"
            "While the round runs you get a live dashboard: [bold]1-9/arrows[/bold] select "
            "an agent, [bold]x[/bold] cancels it (the others keep going), [bold]q[/bold] "
            "stops the whole round, [bold]?[/bold] shows help.",
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
    max_cost: Annotated[
        float | None,
        typer.Option(min=0.0, help="Stop before a new round once model spend (USD) hits this"),
    ] = None,
) -> None:
    """Point the Research Director at a project, lock in goal and baseline, and research."""
    _load_env()
    repo = (repo or Path(".")).resolve()
    settings = SessionSettings()
    session = SetupSession(openrouter_client(), console, settings=settings, repo=repo)
    console.print(f"Project: [bold]{repo}[/bold]")
    console.print(
        Panel(
            "The Research Director surveys your project itself. Set what it needs with "
            "slash commands or just say it - type [bold cyan]/[/bold cyan] to see the "
            "command menu:\n"
            "  [bold cyan]/goal[/bold cyan] reduce wmape        "
            "[bold cyan]/baseline[/bold cyan] models/arima.py\n"
            "Once goal, baseline, and metric are settled it evaluates the baseline, opens "
            "a session folder like research/001-reduce-wmape/, and pops the round-1 plan "
            "open in your editor. You edit it, type execute, and the round runs. Each "
            "round adds round-N-plan.md and round-N-findings.md there, indexed by the "
            "session's README.md - a lab notebook of all the research ever tried.",
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
                session_dir=session.session_dir,
                max_cost=max_cost,
                review=review_round,
            )
        )
    except KeyboardInterrupt:
        console.print(
            f"\nRun interrupted. Resume later with: autoresearch resume -c {session.config_path}"
        )
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
        "Rank",
        "ID",
        "Experiment",
        "Skills",
        metric.upper(),
        "RMSE",
        "Holdout",
        "Cost",
        "Promoted",
    ):
        table.add_column(column)
    for rank, item in enumerate(store.leaderboard(metric, state.get("metric_direction", "min")), 1):
        table.add_row(
            str(rank),
            item.id,
            item.idea.title,
            ", ".join(item.idea.skills_used),
            f"{item.metrics.get(metric, float('nan')):.6f}",
            f"{item.metrics.get('rmse', float('nan')):.3f}",
            f"{item.holdout_metrics.get('wmape', float('nan')):.6f}",
            format_cost(item.metadata.get("cost_usd", 0.0) or 0.0),
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
def doctor(
    no_api: Annotated[
        bool, typer.Option("--no-api", help="Skip the live OpenRouter key check")
    ] = False,
) -> None:
    """Check that tools and credentials needed to run research are in place."""
    _load_env()
    symbols = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✗[/red]"}
    table = Table(title="autoresearch doctor", show_edge=False)
    table.add_column("")
    table.add_column("Check")
    table.add_column("Detail")
    checks = run_checks(check_api=not no_api)
    for check in checks:
        table.add_row(symbols[check.status], check.name, check.detail)
    console.print(table)
    failures = [c for c in checks if c.status == "fail"]
    if failures:
        console.print(
            f"\n[red]{len(failures)} problem(s) must be fixed before a run.[/red] "
            "See the setup steps in the README."
        )
        raise typer.Exit(1)
    console.print("\n[green]Ready to run.[/green]")


@app.command()
def stop() -> None:
    """Stop a running orchestrator: mid-round agents are cancelled within seconds."""
    Path(".autoresearch-stop").write_text("stop\n")
    console.print("Stop requested. Running agents will be cancelled and recorded as such.")


if __name__ == "__main__":
    app()
