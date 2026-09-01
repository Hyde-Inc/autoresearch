from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
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
from .report import build_report, summarize_run
from .skills import load_skills
from .slash import SessionSettings, parse_reply, print_help
from .store import RunStore, latest_run
from .ui import THEME

app = typer.Typer(no_args_is_help=True, help="Parallel autonomous ML experimentation.")
console = Console(theme=THEME)
ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Task YAML")
]


class _State:
    """Global flags shared across commands (set by the app-level callback)."""

    def __init__(self) -> None:
        self.json = False


STATE = _State()


def _emit_json(payload: object) -> None:
    """Print a machine-readable JSON payload (used when ``--json`` is set)."""
    console.print_json(json.dumps(payload, default=str))


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"autoresearch {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the autoresearch version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of tables"),
    ] = False,
) -> None:
    """Parallel autonomous ML experimentation."""
    STATE.json = json_output


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
        console.print(f"[dark_orange3]! {warning}[/dark_orange3]")


def _store_from(run_dir: Path | None, config: Path | None = None) -> RunStore:
    if run_dir:
        return RunStore(run_dir)
    if config:
        cfg = load_config(config)
        found = latest_run(cfg.resolve(cfg.workspace.runs), cfg.name)
    else:
        # `start`-created tasks keep runs under the hidden .autoresearch/runs.
        found = latest_run(Path("runs")) or latest_run(Path(".autoresearch/runs"))
    if found is None:
        raise typer.BadParameter("no run found; pass --run-dir")
    return RunStore(found)


def _print_run_summary(store: RunStore) -> None:
    """End-of-run recap: outcome vs baseline, best attempt, spend, and paths."""
    summary = summarize_run(store.run_dir)
    if STATE.json:
        _emit_json(summary)
        return

    metric = summary["metric"]
    counts = summary["counts"]
    baseline, final = summary["baseline"], summary["final"]
    improvement = summary["improvement_pct"]

    def _fmt(value: object) -> str:
        return f"{value:.6f}" if isinstance(value, (int, float)) else "n/a"

    if improvement is None:
        delta = "[grey62]no baseline comparison[/grey62]"
    else:
        color = "green4" if improvement > 0 else "red3" if improvement < 0 else "grey62"
        delta = f"[{color}]{improvement:+.2f}%[/{color}]"

    lines = [
        (f"[bold]{summary['task']}[/bold]  ·  {counts['completed']} experiments"
         f" over {summary['rounds'] or '?'} round(s)"),
        "",
        f"{metric.upper()}: baseline {_fmt(baseline)} → final {_fmt(final)}  ({delta})",
        (f"Passed {counts['passed']} · promoted {counts['promoted']} · "
         f"rejected {counts['rejected']} · failed {counts['failed']} · "
         f"cancelled {counts['cancelled']}"),
        f"Model spend: {format_cost(summary['total_cost_usd'])}",
    ]
    best = summary["best_attempt"]
    if best is not None:
        star = " [green4]★ promoted[/green4]" if best["promoted"] else ""
        lines.append(f"Best: [bold]{best['title']}[/bold] ({best['id']}) "
                     f"{metric}={_fmt(best['metric'])}{star}")
    lines.extend([
        "",
        f"[grey62]{store.run_dir}[/grey62]",
        "[grey62]autoresearch report · autoresearch leaderboard[/grey62]",
    ])
    console.print(Panel("\n".join(lines), title="Run summary", border_style="grey42"))


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
    _print_run_summary(store)


@app.command(name="foundry-setup")
def foundry_setup(
    repo: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="Path to the Foundry transforms repo"),
    ],
    project_folder_rid: Annotated[
        str, typer.Option(help="Project folder RID (else resolved from the repo's git remote)")
    ] = "",
    branch: Annotated[str, typer.Option(help="Foundry branch to write/build on")] = "master",
    n_skus: Annotated[int, typer.Option(min=2)] = 10,
    n_days: Annotated[int, typer.Option(min=60)] = 210,
    validation_days: Annotated[int, typer.Option(min=1)] = 28,
    holdout_days: Annotated[int, typer.Option(min=1)] = 28,
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            help=(
                "Use an EXISTING dataset instead of provisioning synthetic data, as "
                "key=RID (repeatable). Keys: sales_train, forecast_request, "
                "validation_actuals, holdout_actuals (required); sales_raw, forecasts, "
                "evaluation_metrics (optional - outputs are created if missing)."
            ),
        ),
    ] = None,
    id_col: Annotated[str, typer.Option(help="Item/series id column name")] = "sku_id",
    date_col: Annotated[str, typer.Option(help="Date column name")] = "date",
    target_col: Annotated[str, typer.Option(help="Target column name")] = "units_sold",
    metric_name: Annotated[
        str, typer.Option("--metric", help="Primary metric: wmape, mape, rmse, or bias_pct")
    ] = "wmape",
    write_config: Annotated[
        Path, typer.Option(help="Where to write the runtime: foundry task.yaml")
    ] = Path("foundry-task.yaml"),
    no_push: Annotated[
        bool, typer.Option("--no-push", help="Scaffold and provision but skip the git push")
    ] = False,
) -> None:
    """Install autoresearch into a Foundry transforms repo, end to end.

    Reads the repo's own gradle.properties for its identity, provisions the
    pipeline datasets (or wires into existing ones via --dataset key=RID),
    scaffolds the pipeline transforms + AUTORESEARCH.md contract + curated
    dependencies into the repo, pushes it so Foundry publishes the baseline,
    and writes a ``runtime: foundry`` task.yaml. After this,
    ``autoresearch run -c foundry-task.yaml`` runs the whole loop autonomously.
    """
    import yaml

    from . import foundry_setup as setup
    from .foundry import ensure_dataset, ensure_folder, resolve_parent_folder
    from .foundry_build import FoundryBuildError, push_repo

    _load_env()
    repo = repo.resolve()
    props = setup.read_gradle_properties(repo)
    repo_rid = props.get("transformsRepoRid") or _repo_rid_from_git(repo)
    repo_path = props.get("transformsRepoPath", "")
    if branch == "master":
        branch = props.get("transformsDefaultBranchName", branch)
    if not repo_path:
        raise typer.BadParameter(
            f"{repo}/gradle.properties has no transformsRepoPath - is this a cloned "
            "Foundry transforms repository?"
        )
    project_path = repo_path.rsplit("/", 1)[0]
    datasets_path = f"{project_path}/{setup.FOLDER}"
    project = project_folder_rid or (resolve_parent_folder(repo_rid) if repo_rid else "")
    if not project:
        raise typer.BadParameter("could not resolve the project folder; pass --project-folder-rid")

    if dataset:
        # Existing-data mode: no synthetic generation, no uploads. Wire the
        # transforms straight to the given RIDs; only missing outputs are created.
        rids: dict[str, str] = {}
        for entry in dataset:
            key, _, rid = entry.partition("=")
            if key not in setup.DATASET_NAMES or not rid:
                raise typer.BadParameter(
                    f"--dataset '{entry}' must be key=RID with key one of "
                    f"{', '.join(setup.DATASET_NAMES)}"
                )
            rids[key] = rid
        required = ("sales_train", "forecast_request", "validation_actuals", "holdout_actuals")
        missing = [key for key in required if key not in rids]
        if missing:
            raise typer.BadParameter(f"--dataset missing required keys: {', '.join(missing)}")
        try:
            folder = ensure_folder(setup.FOLDER, project)
            for key in ("forecasts", "evaluation_metrics"):
                if key not in rids:
                    rids[key] = ensure_dataset(setup.DATASET_NAMES[key], folder)
                    console.print(f"  created output dataset {key}: {rids[key]}")
        except FoundryError as exc:
            console.print(f"[red]Foundry setup failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        datasets_config = {key: rids.get(key, "") for key in setup.CONFIG_DATASET_KEYS}
        refs = dict(rids)
        data_home = "existing project datasets, wired by RID"
    else:
        console.print("[bold grey19]Generating synthetic demand history[/bold grey19]")
        history = setup.generate_history(n_skus=n_skus, n_days=n_days)
        frames = setup.chronological_split(
            history, validation_days=validation_days, holdout_days=holdout_days
        )
        # The raw feed is the dirty version of the train split: the on-Foundry
        # data_preprocessing stage cleans it back into sales_train.
        frames["sales_raw"] = setup.make_raw(frames["sales_train"])
        summary = Table(title="Chronological split", show_edge=False)
        summary.add_column("Split")
        summary.add_column("Rows", justify="right")
        for key in ("sales_raw", "validation_actuals", "holdout_actuals", "forecast_request"):
            summary.add_row(key, f"{len(frames[key]):,}")
        console.print(summary)

        console.print(f"[bold grey19]Provisioning datasets under[/bold grey19] {datasets_path}")
        try:
            result = setup.provision(frames, project_folder_rid=project, branch=branch)
        except FoundryError as exc:
            console.print(f"[red]Foundry setup failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        datasets_config = setup.result_as_config(result)
        refs = setup.default_refs(datasets_path)
        data_home = f"under `{datasets_path}/`"

    table = Table(title="Foundry datasets", show_edge=False)
    table.add_column("Dataset")
    table.add_column("RID")
    for key, rid in datasets_config.items():
        if rid:
            table.add_row(key, rid)
    console.print(table)

    console.print("[bold grey19]Scaffolding the repo[/bold grey19]")
    transform_rel, actions = setup.scaffold(
        repo,
        refs,
        branch,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        data_home=data_home,
    )
    for action in actions:
        console.print(f"  {action}")

    if no_push:
        console.print("[dark_orange3]--no-push: remember to push before running[/dark_orange3]")
    else:
        console.print(f"[bold grey19]Pushing to Foundry[/bold grey19] ({branch}) so the baseline publishes")
        try:
            push_repo(repo, branch=branch, message="autoresearch: install baseline transform")
        except FoundryBuildError as exc:
            console.print(f"[red]push failed:[/red] {exc}")
            raise typer.Exit(1) from exc

    task = {
        "name": "foundry-demand-forecasting",
        "goal": f"Reduce validation {metric_name.upper()} on the Foundry demand dataset.",
        "runtime": "foundry",
        "metric": {"name": metric_name, "direction": "min"},
        "data": {"id_column": id_col, "date_column": date_col, "target_column": target_col},
        # Foundry sessions are slower per iteration (publish + build) and the
        # transforms repos are larger to explore, so give coding sessions more
        # headroom than the local defaults while keeping spend bounded.
        "agents": {"count": 3, "timeout_s": 1500, "budget_s": 3000},
        "budget": {"rounds": 3, "max_experiments": 9},
        "workspace": {
            "seed": str(repo),
            "runs": "runs",
            "allowed_paths": [transform_rel],
        },
        "foundry": {
            "repo_dir": str(repo),
            "branch": branch,
            "repo_rid": repo_rid,
            "project_folder_rid": project,
            "datasets": datasets_config,
        },
    }
    write_config = write_config.resolve()
    write_config.write_text(yaml.safe_dump(task, sort_keys=False))
    console.print(f"\n[green]Wrote task config:[/green] {write_config}")
    console.print(f"Next: [bold grey19]autoresearch run -c {write_config}[/bold grey19]")


def _repo_rid_from_git(repo: Path) -> str:
    """Best-effort: read the Stemma repo RID from the git remote URL."""
    import re
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo), "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"(ri\.stemma\.main\.repository\.[0-9a-fA-F-]+)", result.stdout)
    return match.group(1) if match else ""


@app.command(name="foundry-build")
def foundry_build_cmd(
    config: ConfigOption,
    message: Annotated[
        str, typer.Option("--message", "-m", help="Commit message for the pushed change")
    ] = "autoresearch: trigger training build",
    no_push: Annotated[
        bool, typer.Option("--no-push", help="Skip git push; build the already-published code")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even if Foundry considers the output up to date")
    ] = False,
    build_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Build the whole pipeline (preprocessing -> model -> evaluation), not just forecasts",
        ),
    ] = False,
    no_wait: Annotated[
        bool,
        typer.Option(
            "--no-wait",
            help="Trigger the build and return immediately instead of polling to completion",
        ),
    ] = False,
    timeout_s: Annotated[
        int | None, typer.Option(min=1, help="Override the build timeout (seconds)")
    ] = None,
) -> None:
    """Push the transforms repo and trigger a Foundry build of the forecasts dataset.

    This is the manual 'train on Foundry' step: it publishes the current transform
    code and runs a build on Foundry compute, then reports the result. With
    ``--all`` the build covers every pipeline output (Foundry runs the jobs in
    dependency order): sales_train, forecasts, evaluation_metrics.
    """
    from .foundry_build import (
        FoundryBuildError,
        create_build_when_ready,
        push_repo,
        wait_for_build,
    )

    _load_env()
    cfg = load_config(config)
    if cfg.runtime != "foundry" or cfg.foundry is None:
        raise typer.BadParameter("config is not runtime: foundry (run foundry-setup first)")
    fdry = cfg.foundry
    if not fdry.datasets.forecasts:
        raise typer.BadParameter("foundry.datasets.forecasts RID is unset; run foundry-setup first")
    targets = fdry.datasets.pipeline_targets() if build_all else [fdry.datasets.forecasts]
    label = "full pipeline" if build_all else "forecasts"
    repo_dir = cfg.resolve(fdry.repo_dir)

    try:
        if not no_push:
            console.print(f"[bold grey19]Pushing[/bold grey19] {repo_dir} → Foundry branch [blue]{fdry.branch}[/blue]")
            sha = push_repo(repo_dir, branch=fdry.branch, message=message)
            console.print(f"  pushed{f' commit {sha[:8]}' if sha else ' (nothing new to commit)'}")
        console.print(f"[bold grey19]Triggering build[/bold grey19] of {label} ({len(targets)} targets)")

        def on_wait(elapsed: float) -> None:
            console.print(f"  [{elapsed:6.0f}s] waiting for Foundry to publish the transform…")

        build_rid = create_build_when_ready(
            targets, branch=fdry.branch, force=force, on_wait=on_wait
        )
        console.print(f"  build {build_rid}")
        if no_wait:
            hostname = os.getenv("FOUNDRY_HOSTNAME", "").strip()
            hostname = hostname.removeprefix("https://").removeprefix("http://").rstrip("/")
            if hostname:
                console.print(
                    "  watch it live: "
                    f"https://{hostname}/workspace/data-integration/dataset/preview/"
                    f"{fdry.datasets.forecasts}/{fdry.branch}"
                )
            console.print("[green4]Build triggered[/green4] — running on Foundry now")
            return

        def on_status(status: str, elapsed: float) -> None:
            console.print(f"  [{elapsed:6.0f}s] {status}")

        result = wait_for_build(
            build_rid,
            timeout_s=timeout_s or fdry.build_timeout_s,
            poll_s=fdry.poll_s,
            on_status=on_status,
        )
        console.print(f"[green]Build {result.status}[/green] — {label} built")
    except (FoundryBuildError, FoundryError) as exc:
        console.print(f"[red]Foundry build failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command(name="foundry-score")
def foundry_score_cmd(
    config: ConfigOption,
    holdout: Annotated[
        bool, typer.Option("--holdout", help="Also score the sealed holdout split")
    ] = False,
) -> None:
    """Read the forecasts + sealed actuals back from Foundry and print the metrics."""
    from .harness import forecasting_metrics

    _load_env()
    cfg = load_config(config)
    if cfg.runtime != "foundry" or cfg.foundry is None:
        raise typer.BadParameter("config is not runtime: foundry")
    fdry = cfg.foundry
    d = cfg.data
    keys = [d.id_column, d.date_column]

    forecasts = read_dataset(fdry.datasets.forecasts, branch=fdry.branch)
    forecasts[d.date_column] = pd.to_datetime(forecasts[d.date_column], utc=True).dt.tz_localize(None)

    def score(rid: str, label: str) -> dict[str, object]:
        actuals = read_dataset(rid, branch=fdry.branch)
        actuals[d.date_column] = pd.to_datetime(actuals[d.date_column], utc=True).dt.tz_localize(None)
        merged = actuals.merge(forecasts[[*keys, "forecast"]], on=keys, validate="one_to_one")
        met = forecasting_metrics(
            merged[d.target_column].to_numpy(float), merged["forecast"].to_numpy(float)
        )
        return {"split": label, "rows": len(merged), **met}

    results = [score(fdry.datasets.validation_actuals, "validation")]
    if holdout:
        results.append(score(fdry.datasets.holdout_actuals, "holdout"))

    if STATE.json:
        _emit_json(results)
        return

    table = Table(title="Foundry evaluation", show_edge=False)
    table.add_column("Split")
    table.add_column("Rows", justify="right")
    table.add_column(cfg.metric.name, justify="right")
    table.add_column("mape", justify="right")
    table.add_column("rmse", justify="right")
    table.add_column("bias%", justify="right")
    for r in results:
        table.add_row(
            str(r["split"]),
            f"{r['rows']:,}",
            f"{r['wmape']:.4f}",
            f"{r['mape']:.4f}",
            f"{r['rmse']:.3f}",
            f"{r['bias_pct']:.2f}",
        )
    console.print(table)


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
    _print_run_summary(completed)


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
        console.print(f"Reading Foundry dataset [bold grey19]{rid}[/bold grey19]" + (f" ({branch})" if branch else ""))
        try:
            frame = read_dataset(rid, branch=branch)
        except FoundryError as exc:
            console.print(f"[red]x {exc}[/red]")
            raise typer.Exit(1) from None
        console.print(f"Downloaded [bold grey19]{len(frame)}[/bold grey19] rows from Foundry")
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
        f"Found [bold grey19]{preview.rows}[/bold grey19] clean rows, [bold grey19]{preview.skus}[/bold grey19] items, "
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
    for column in ("#", "Experiment", "Hypothesis", "Why", "Skills"):
        table.add_column(column)
    for index, idea in enumerate(ideas, 1):
        why = "; ".join(item.observation for item in idea.evidence[:2])
        table.add_row(
            str(index),
            idea.title,
            idea.hypothesis,
            why or "-",
            ", ".join(idea.skills_used) or "-",
        )
    console.print(table)
    console.print(
        Panel(
            f"[bold grey19]{plan_path}[/bold grey19] (opened in your editor)\n"
            "Edit the file freely - the edited file is exactly what runs.\n"
            "Reply [green]execute[/green] to launch, [red]stop[/red] to end the run, or "
            "give feedback to revise the plan.\n"
            "While the round runs you get a live dashboard: [bold grey19]1-9/arrows[/bold grey19] select "
            "an agent, [bold grey19]x[/bold grey19] cancels it (the others keep going), [bold grey19]q[/bold grey19] "
            "stops the whole round, [bold grey19]?[/bold grey19] shows help.",
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
    console.print(f"Project: [bold grey19]{repo}[/bold grey19]")
    console.print(
        Panel(
            "The Research Director surveys your project itself. Set what it needs with "
            "slash commands or just say it - type [bold blue]/[/bold blue] to see the "
            "command menu:\n"
            "  [bold blue]/goal[/bold blue] reduce wmape        "
            "[bold blue]/baseline[/bold blue] models/arima.py\n"
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
    console.print(f"Run complete: [bold grey19]{store.run_dir}[/bold grey19]")


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
        console.print(f"[dark_orange3]Guardrail failures:[/dark_orange3] {result.guardrail_failures}")


@app.command()
def leaderboard(
    run_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    config: Annotated[Path | None, typer.Option("-c", exists=True, dir_okay=False)] = None,
) -> None:
    """Show ranked, guardrail-passing attempts."""
    store = _store_from(run_dir, config)
    state = store.load_state()
    metric = state.get("primary_metric", "wmape")
    ranked = store.leaderboard(metric, state.get("metric_direction", "min"))
    if STATE.json:
        _emit_json(
            [
                {
                    "rank": rank,
                    "id": item.id,
                    "experiment": item.idea.title,
                    "skills": item.idea.skills_used,
                    metric: item.metrics.get(metric),
                    "rmse": item.metrics.get("rmse"),
                    "holdout_wmape": item.holdout_metrics.get("wmape"),
                    "promoted": item.promoted,
                }
                for rank, item in enumerate(ranked, 1)
            ]
        )
        return
    table = Table(title=f"{state.get('task', 'Autoresearch')} leaderboard")
    for column in (
        "Rank",
        "ID",
        "Experiment",
        "Skills",
        metric.upper(),
        "RMSE",
        "Holdout",
        "Promoted",
    ):
        table.add_column(column)
    for rank, item in enumerate(ranked, 1):
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
def doctor(
    no_api: Annotated[
        bool, typer.Option("--no-api", help="Skip the live OpenRouter key check")
    ] = False,
) -> None:
    """Check that tools and credentials needed to run research are in place."""
    _load_env()
    checks = run_checks(check_api=not no_api)
    failures = [c for c in checks if c.status == "fail"]
    if STATE.json:
        _emit_json(
            {
                "ready": not failures,
                "checks": [
                    {"name": c.name, "status": c.status, "detail": c.detail} for c in checks
                ],
            }
        )
        if failures:
            raise typer.Exit(1)
        return
    symbols = {
        "ok": "[green4]✓[/green4]",
        "warn": "[dark_orange3]![/dark_orange3]",
        "fail": "[red3]✗[/red3]",
    }
    table = Table(title="autoresearch doctor", show_edge=False)
    table.add_column("")
    table.add_column("Check")
    table.add_column("Detail")
    for check in checks:
        table.add_row(symbols[check.status], check.name, check.detail)
    console.print(table)
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
