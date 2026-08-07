from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .config import load_config
from .ingest import ingest_csv, preview_csv
from .metrics import MetricInterpreter, MetricSpec, MetricValidation, adopt_spec, eval_columns
from .orchestrator import run_research, validate_baseline
from .report import build_report
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
    csv: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Sales CSV to ingest")],
    name: Annotated[str, typer.Option("--name", "-n", help="Task name")],
    tasks_root: Annotated[Path, typer.Option(help="Where task folders are created")] = Path("tasks"),
    validation_days: Annotated[int | None, typer.Option(help="Validation horizon (days)")] = None,
    holdout_days: Annotated[int | None, typer.Option(help="Hidden holdout horizon (days)")] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing task of the same name")] = False,
) -> None:
    """Ingest a standard-schema CSV (date, sku_name, sales, selling_price) into a task."""
    preview = preview_csv(csv)
    for warning in preview.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    if not preview.ok:
        for error in preview.errors:
            console.print(f"[red]✗ {error}[/red]")
        raise typer.Exit(1)
    console.print(
        f"Parsed [bold]{preview.rows}[/bold] rows · {preview.skus} SKUs · "
        f"{preview.distinct_dates} dates ({preview.date_min} → {preview.date_max})"
    )
    result = ingest_csv(
        csv,
        name=name,
        tasks_root=tasks_root,
        validation_days=validation_days,
        holdout_days=holdout_days,
        overwrite=overwrite,
    )
    console.print(
        f"[green]Created task[/green] [bold]{result.task}[/bold] at {result.task_dir}\n"
        f"  train={result.train_rows} rows (→ {result.train_end}), "
        f"validation={result.validation_rows} (→ {result.validation_end}), "
        f"holdout={result.holdout_rows} (→ {result.holdout_end})\n"
        f"  next: [bold]autoresearch metric -c {result.config_path}[/bold] \"your metric\""
    )


def _print_spec(spec: MetricSpec, validation: MetricValidation) -> None:
    console.print(f"\n[bold]Metric:[/bold] {spec.name} ({spec.direction}imize)")
    console.print(f"[bold]Understanding:[/bold] {spec.understanding}\n")
    table = Table(title="Hand-worked example")
    columns = list(spec.example.rows[0].keys())
    for column in columns:
        table.add_column(column)
    for row in spec.example.rows:
        table.add_row(*(str(row.get(column, "")) for column in columns))
    console.print(table)
    for step in spec.example.steps:
        console.print(f"  • {step}")
    console.print(f"[bold]Hand-computed value:[/bold] {spec.example.value}\n")
    console.print("[bold]Grader code:[/bold]")
    console.print(spec.code, markup=False, highlight=True)
    console.print()
    for check in validation.checks:
        color = "green" if check.passed else "red"
        console.print(f"[{color}]✓ {check.name}[/{color}]: {check.detail}")
    for warning in validation.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")


@app.command()
def metric(
    config: ConfigOption,
    description: Annotated[
        str, typer.Argument(help="Plain-English description of the evaluation metric")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Define the grader's evaluation metric from a plain-language description."""
    load_dotenv()
    cfg = load_config(config)
    columns = eval_columns(cfg)
    console.print(
        f"Interpreting metric with [bold]{cfg.director.model}[/bold] "
        f"(eval columns: {', '.join(columns)})..."
    )
    interpreter = MetricInterpreter(cfg.director.model, cfg.director.temperature)
    spec, validation = asyncio.run(interpreter.interpret(description, columns))
    _print_spec(spec, validation)
    if not yes and not typer.confirm("Adopt this metric as the grader?"):
        console.print("Metric discarded; nothing was changed.")
        raise typer.Exit(0)
    saved = adopt_spec(cfg, spec)
    console.print(
        f"\n[green]Adopted.[/green] Spec saved to [bold]{saved}[/bold]; "
        f"task.yaml now grades on [bold]{spec.name}[/bold] ({spec.direction})."
    )


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
    for column in ("Rank", "ID", "Experiment", metric.upper(), "RMSE", "Holdout", "Promoted"):
        table.add_column(column)
    for rank, item in enumerate(
        store.leaderboard(metric, state.get("metric_direction", "min")), 1
    ):
        table.add_row(
            str(rank),
            item.id,
            item.idea.title,
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


@app.command()
def ui(
    runs_dir: Annotated[Path, typer.Option(help="Runs directory to serve")] = Path("runs"),
    port: Annotated[int, typer.Option(help="Port for the dashboard")] = 8500,
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
) -> None:
    """Launch the web dashboard."""
    import uvicorn

    from .webapp import create_app

    load_dotenv()
    console.print(f"Dashboard: [bold]http://{host}:{port}[/bold]")
    uvicorn.run(create_app(runs_dir, Path.cwd()), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
