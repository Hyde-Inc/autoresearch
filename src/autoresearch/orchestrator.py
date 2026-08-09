from __future__ import annotations

import asyncio
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from .config import TaskConfig
from .director import ResearchDirector
from .harness import Evaluation, evaluate
from .models import Attempt, Idea
from .store import RunStore
from .usage import log_usage
from .worker import WorkerResult, _run, ensure_seed_repo, remove_worktree, run_worker

console = Console()

# Demo hard caps: no run may exceed these regardless of config or API input.
DEMO_MAX_ROUNDS = 4
DEMO_MAX_PARALLEL = 4


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate < incumbent if direction == "min" else candidate > incumbent


def _paths_allowed(paths: list[str], prefixes: list[str]) -> list[str]:
    return [path for path in paths if not any(path.startswith(prefix) for prefix in prefixes)]


async def validate_baseline(config: TaskConfig) -> Evaluation:
    seed = config.resolve(config.workspace.seed)
    return await evaluate(seed, config, baseline=None, guardrails=config.guardrails)


async def _execute_attempt(
    config: TaskConfig,
    store: RunStore,
    repo: Path,
    idea: Idea,
    attempt_id: str,
    round_number: int,
    base_ref: str,
    baseline: dict[str, float],
) -> tuple[Attempt, WorkerResult]:
    attempt = Attempt(id=attempt_id, round=round_number, idea=idea, status="running")
    store.save_attempt(attempt)
    worker = await run_worker(
        config=config,
        store=store,
        repo=repo,
        idea=idea,
        attempt_id=attempt_id,
        base_ref=base_ref,
    )
    attempt.branch = worker.branch
    attempt.commit = worker.commit
    attempt.log_path = str(store.logs_dir / f"{attempt_id}.jsonl")
    diff_path = store.attempts_dir / f"{attempt_id}.patch"
    diff_path.write_text(worker.diff)
    attempt.diff_path = str(diff_path)
    forbidden = _paths_allowed(worker.changed_paths, config.workspace.allowed_paths)
    if forbidden:
        attempt.status = "rejected"
        attempt.error = f"agent changed forbidden paths: {forbidden}"
    elif worker.error:
        attempt.status = "failed"
        attempt.error = worker.error
    else:
        result = await evaluate(worker.worktree, config, baseline=baseline)
        attempt.metrics = result.metrics
        attempt.holdout_metrics = result.holdout_metrics
        attempt.duration_s = result.duration_s
        attempt.guardrail_failures = result.guardrail_failures
        attempt.error = result.error
        attempt.status = "passed" if result.passed else ("failed" if result.error else "rejected")
    store.save_attempt(attempt)
    return attempt, worker


async def run_research(
    config: TaskConfig,
    *,
    goal: str | None = None,
    guardrails: list[str] | None = None,
    parallel: int | None = None,
    max_experiments: int | None = None,
    rounds: int | None = None,
    round_timeout_s: int | None = None,
    cost_limit_usd: float | None = None,
    tracks: list[str] | None = None,
    resume_store: RunStore | None = None,
) -> RunStore:
    if goal:
        config.goal = goal
    if guardrails:
        config.guardrails.extend(guardrails)
    tracks = [t for t in (tracks or [])]
    parallel = min(len(tracks) or parallel or config.agents.count, DEMO_MAX_PARALLEL)
    tracks = tracks[:parallel]
    max_rounds = min(rounds or config.budget.rounds, DEMO_MAX_ROUNDS)
    # Rounds (not experiments) is the primary budget control; derive the experiment
    # ceiling from rounds unless an explicit cap was passed.
    maximum = max_experiments or (parallel * max_rounds)
    if round_timeout_s:
        # A per-round time budget: agents in a round run in parallel, so bounding each
        # agent's wall-clock effectively bounds the round.
        config.agents.timeout_s = round_timeout_s
    run_config = {
        "agents": parallel,
        "rounds": max_rounds,
        "round_timeout_s": config.agents.timeout_s,
        "cost_limit_usd": cost_limit_usd,
        "tracks": tracks,
    }
    seed_template = config.resolve(config.workspace.seed)
    if not seed_template.exists():
        raise RuntimeError(f"seed workspace does not exist: {seed_template}; run prepare.py first")
    if resume_store:
        store = resume_store
        repo = store.run_dir / "repo"
        if not (repo / ".git").exists():
            raise RuntimeError(f"run repository is missing: {repo}")
        prior = store.load_state()
        baseline = prior["baseline"]
        incumbent = prior["incumbent"]
        incumbent_ref = prior.get("incumbent_ref", "main")
        completed = int(prior.get("completed", len(store.load_attempts())))
        round_number = int(prior.get("round", 0))
        created_at = prior.get("created_at", datetime.now(UTC).isoformat())
        total_cost = float(prior.get("total_cost_usd", 0.0))
        run_config = prior.get("run_config", run_config)
    else:
        store = RunStore.create(config.resolve(config.workspace.runs), config.name)
        shutil.copy2(config.config_path, store.run_dir / "task.yaml")
        repo = store.run_dir / "repo"
        shutil.copytree(seed_template, repo)
        await ensure_seed_repo(repo)
        created_at = datetime.now(UTC).isoformat()
        total_cost = 0.0
        baseline_eval = await validate_baseline(config)
        if baseline_eval.error:
            raise RuntimeError(f"baseline evaluation failed: {baseline_eval.error}")
        baseline = dict(baseline_eval.metrics)
        holdout_key = f"holdout_{config.metric.name}"
        baseline[holdout_key] = baseline_eval.holdout_metrics[config.metric.name]
        incumbent = baseline
        incumbent_ref = "main"
        completed = 0
        round_number = 0

    def persist(status: str) -> None:
        store.save_state(
            {
                "task": config.name,
                "status": status,
                "goal": config.goal,
                "primary_metric": config.metric.name,
                "metric_direction": config.metric.direction,
                "baseline": baseline,
                "incumbent": incumbent,
                "incumbent_ref": incumbent_ref,
                "round": round_number,
                "completed": completed,
                "created_at": created_at,
                "run_config": run_config,
                "total_cost_usd": round(total_cost, 4),
            }
        )

    if not resume_store:
        persist("running")
    console.print(
        f"[bold]Baseline[/bold] {config.metric.name}={baseline[config.metric.name]:.6f}, "
        f"holdout={baseline[f'holdout_{config.metric.name}']:.6f}"
    )
    director = ResearchDirector(config)
    try:
        while completed < maximum and round_number < max_rounds:
            if Path(".autoresearch-stop").exists():
                break
            round_number += 1
            count = min(parallel, maximum - completed)
            round_tracks = tracks[:count] if tracks else None
            notes = store.notes_file.read_text() if store.notes_file.exists() else ""
            ideas = await director.propose(
                count=count,
                round_number=round_number,
                attempts=store.load_attempts(),
                notes=notes,
                tracks=round_tracks,
            )
            console.print(f"[bold cyan]Round {round_number}[/bold cyan]: launching {count} experiments")
            jobs = []
            for idea in ideas:
                attempt_id = uuid.uuid4().hex[:8]
                jobs.append(
                    _execute_attempt(
                        config,
                        store,
                        repo,
                        idea,
                        attempt_id,
                        round_number,
                        incumbent_ref,
                        incumbent,
                    )
                )
            results = await asyncio.gather(*jobs)
            completed += len(results)
            passing = [
                (attempt, worker)
                for attempt, worker in results
                if attempt.status == "passed" and config.metric.name in attempt.metrics
            ]
            passing.sort(
                key=lambda pair: pair[0].metrics[config.metric.name],
                reverse=config.metric.direction == "max",
            )
            winner = passing[0] if passing else None
            if winner and _better(
                winner[0].metrics[config.metric.name],
                incumbent[config.metric.name],
                config.metric.direction,
            ):
                attempt, worker = winner
                code, output = await _run("git", "merge", "--ff-only", worker.branch, cwd=repo)
                if code:
                    attempt.status = "failed"
                    attempt.error = f"promotion failed: {output}"
                else:
                    attempt.promoted = True
                    incumbent_ref = "main"
                    incumbent = dict(attempt.metrics)
                    incumbent[f"holdout_{config.metric.name}"] = attempt.holdout_metrics[
                        config.metric.name
                    ]
                    console.print(
                        f"[green]Promoted {attempt.id}[/green]: "
                        f"{config.metric.name}={attempt.metrics[config.metric.name]:.6f}"
                    )
                store.save_attempt(attempt)
            for attempt, _ in results:
                if attempt.log_path:
                    total_cost += log_usage(Path(attempt.log_path))[1]
            reflection = await director.reflect([attempt for attempt, _ in results])
            store.append_note(f"Round {round_number}", reflection)
            for _, worker in results:
                await remove_worktree(repo, worker)
            persist("running")
            if cost_limit_usd is not None and total_cost >= cost_limit_usd:
                store.append_note(
                    "Budget",
                    f"Stopped after round {round_number}: cost ${total_cost:.2f} reached the "
                    f"${cost_limit_usd:.2f} limit.",
                )
                console.print(
                    f"[yellow]Cost limit reached[/yellow]: ${total_cost:.2f} >= ${cost_limit_usd:.2f}"
                )
                break
    finally:
        stopped = Path(".autoresearch-stop").exists()
        persist("stopped" if stopped else "completed")
        Path(".autoresearch-stop").unlink(missing_ok=True)
    return store
