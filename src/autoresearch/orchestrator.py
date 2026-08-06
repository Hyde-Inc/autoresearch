from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from rich.console import Console

from .config import TaskConfig
from .director import ResearchDirector
from .harness import Evaluation, evaluate
from .models import Attempt, Idea
from .store import RunStore
from .worker import WorkerResult, _run, ensure_seed_repo, remove_worktree, run_worker

console = Console()


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
    resume_store: RunStore | None = None,
) -> RunStore:
    if goal:
        config.goal = goal
    if guardrails:
        config.guardrails.extend(guardrails)
    parallel = parallel or config.agents.count
    maximum = max_experiments or config.budget.max_experiments
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
    else:
        store = RunStore.create(config.resolve(config.workspace.runs), config.name)
        shutil.copy2(config.config_path, store.run_dir / "task.yaml")
        repo = store.run_dir / "repo"
        shutil.copytree(seed_template, repo)
        await ensure_seed_repo(repo)
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
        store.save_state(
            {
                "task": config.name,
                "status": "running",
                "goal": config.goal,
                "primary_metric": config.metric.name,
                "metric_direction": config.metric.direction,
                "baseline": baseline,
                "incumbent": baseline,
                "incumbent_ref": "main",
                "completed": 0,
                "round": 0,
            }
        )
    console.print(
        f"[bold]Baseline[/bold] {config.metric.name}={baseline[config.metric.name]:.6f}, "
        f"holdout={baseline[f'holdout_{config.metric.name}']:.6f}"
    )
    director = ResearchDirector(config)
    try:
        while completed < maximum and round_number < config.budget.rounds:
            if Path(".autoresearch-stop").exists():
                break
            round_number += 1
            count = min(parallel, maximum - completed)
            notes = store.notes_file.read_text() if store.notes_file.exists() else ""
            ideas = await director.propose(
                count=count,
                round_number=round_number,
                attempts=store.load_attempts(),
                notes=notes,
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
            reflection = await director.reflect([attempt for attempt, _ in results])
            store.append_note(f"Round {round_number}", reflection)
            for _, worker in results:
                await remove_worktree(repo, worker)
            store.save_state(
                {
                    "task": config.name,
                    "status": "running",
                    "goal": config.goal,
                    "primary_metric": config.metric.name,
                    "metric_direction": config.metric.direction,
                    "baseline": baseline,
                    "incumbent": incumbent,
                    "incumbent_ref": incumbent_ref,
                    "round": round_number,
                    "completed": completed,
                }
            )
    finally:
        state = store.load_state()
        state["status"] = "stopped" if Path(".autoresearch-stop").exists() else "completed"
        store.save_state(state)
        Path(".autoresearch-stop").unlink(missing_ok=True)
    return store
