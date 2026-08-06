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


if __name__ == "__main__":
    app()
