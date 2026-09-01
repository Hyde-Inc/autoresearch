from __future__ import annotations

import asyncio
import contextlib
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console

from .agent_chat import RoundChat
from .agent_status import (
    CANCELLED,
    EVALUATING,
    FAILED,
    PASSED,
    REJECTED,
    AgentStatus,
)
from .config import TaskConfig
from .costs import format_cost, log_cost
from .dashboard import AgentDashboard
from .director import ResearchDirector, openrouter_client
from .editor import open_in_editor
from .harness import Evaluation, evaluate
from .harness_foundry import evaluate as foundry_evaluate
from .models import Attempt, Idea
from .plans import (
    PlanError,
    mark_executed,
    new_session_dir,
    parse_plan,
    plan_path,
    plans_dir_for_task,
    refresh_session_readme,
    render_plan,
    write_findings,
)
from .progress import Activity
from .store import RunStore
from .ui import THEME
from .worker import (
    WorkerResult,
    _run,
    ensure_seed_repo,
    remove_worktree,
    remove_worktree_path,
    run_worker,
)

console = Console(theme=THEME)

PROPOSALS_MIN = 2
PROPOSALS_MAX = 5


@dataclass
class ReviewDecision:
    """A human reviewer's verdict on the director's proposed round."""

    action: Literal["approve", "revise", "stop"]
    feedback: str | None = None


ReviewCallback = Callable[[int, list[Idea], Path], ReviewDecision]


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate < incumbent if direction == "min" else candidate > incumbent


def _paths_allowed(paths: list[str], prefixes: list[str]) -> list[str]:
    return [path for path in paths if not any(path.startswith(prefix) for prefix in prefixes)]


def _ignore_runs_root(runs_root: Path) -> Callable[[str, list[str]], set[str]]:
    """``copytree`` ignore callback that skips the runs directory itself.

    When the runs folder lives inside the seed repo (e.g. the task.yaml sits in
    the repo and ``workspace.runs`` is relative), a plain copytree would copy
    the run into itself recursively until the OS path-length limit."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if (Path(directory) / name).resolve() == runs_root}

    return ignore


def _main_ref(config: TaskConfig) -> str:
    """The branch experiments start from and winners merge back into."""
    if config.runtime == "foundry" and config.foundry is not None:
        return config.foundry.branch
    return "main"


async def validate_baseline(config: TaskConfig) -> Evaluation:
    seed = config.resolve(config.workspace.seed)
    if config.runtime == "foundry":
        # Baseline = whatever transform is currently published on the main
        # branch; never push local state from the user's repo.
        return await foundry_evaluate(
            seed,
            config,
            baseline=None,
            guardrails=config.guardrails,
            push=False,
            branch=_main_ref(config),
        )
    return await evaluate(seed, config, baseline=None, guardrails=config.guardrails)


async def _final_evaluate(
    worktree: Path,
    config: TaskConfig,
    baseline: dict[str, float],
    on_progress,
) -> Evaluation:
    """Final gate (validation + holdout), routed by runtime. On Foundry the
    experiment's branch was already built during the worker loop, so this
    usually only re-reads the forecasts (it rebuilds if the worker rolled back
    to an earlier snapshot)."""
    if config.runtime == "foundry":
        return await foundry_evaluate(worktree, config, baseline=baseline, on_progress=on_progress)
    return await evaluate(worktree, config, baseline=baseline, on_progress=on_progress)


async def _execute_attempt(
    config: TaskConfig,
    store: RunStore,
    repo: Path,
    idea: Idea,
    attempt_id: str,
    round_number: int,
    base_ref: str,
    baseline: dict[str, float],
    status: AgentStatus | None = None,
) -> tuple[Attempt, WorkerResult | None]:
    """Run one agent end to end, updating its dashboard row along the way.

    Cancelling the surrounding task (dashboard ``x``/``q`` or the stop flag)
    kills the agent's subprocesses, records the attempt as ``cancelled``,
    cleans its worktree, and returns normally so sibling agents keep running.
    """
    tracker = status or AgentStatus(index=0, attempt_id=attempt_id, title=idea.title)
    attempt = Attempt(id=attempt_id, round=round_number, idea=idea, status="running")
    attempt.log_path = str(store.logs_dir / f"{attempt_id}.jsonl")
    tracker.log_path = store.logs_dir / f"{attempt_id}.jsonl"
    store.save_attempt(attempt)
    worker: WorkerResult | None = None
    try:
        worker = await run_worker(
            config=config,
            store=store,
            repo=repo,
            idea=idea,
            attempt_id=attempt_id,
            base_ref=base_ref,
            baseline=baseline,
            on_event=tracker.apply_event,
            on_phase=tracker.set,
        )
        attempt.branch = worker.branch
        attempt.commit = worker.commit
        attempt.metadata["sessions"] = worker.sessions
        attempt.metadata["loop"] = worker.history
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
            tracker.set(EVALUATING, "final gate: validation + holdout splits")
            result = await _final_evaluate(
                worker.worktree,
                config,
                baseline,
                lambda line: tracker.set(action=line),
            )
            attempt.metrics = result.metrics
            attempt.holdout_metrics = result.holdout_metrics
            attempt.duration_s = result.duration_s
            attempt.guardrail_failures = result.guardrail_failures
            attempt.error = result.error
            attempt.status = (
                "passed" if result.passed else ("failed" if result.error else "rejected")
            )
            if result.validation_frame is not None:
                store.save_validation_frame(attempt_id, result.validation_frame)
        primary = attempt.metrics.get(config.metric.name)
        outcome = f"{config.metric.name} {primary:.4f}" if primary is not None else attempt.error
        tracker.finish(
            {"passed": PASSED, "failed": FAILED}.get(attempt.status, REJECTED),
            outcome or attempt.status,
        )
    except asyncio.CancelledError:
        # Swallow the cancellation so the round's gather keeps sibling results.
        task = asyncio.current_task()
        while task is not None and task.cancelling():
            task.uncancel()
        attempt.status = "cancelled"
        attempt.error = "cancelled by user"
        tracker.finish(CANCELLED, "cancelled by user")
        if worker is None:
            # run_worker never returned, but its worktree may already exist.
            await remove_worktree_path(repo, store.worktrees_dir / attempt_id)
    finally:
        cost = log_cost(store.logs_dir / f"{attempt_id}.jsonl")
        attempt.metadata["cost_usd"] = round(cost.cost_usd, 6)
        attempt.metadata["tokens"] = {
            "input": cost.input_tokens,
            "output": cost.output_tokens,
            "reasoning": cost.reasoning_tokens,
        }
        tracker.cost_usd = cost.cost_usd
        store.save_attempt(attempt)
    return attempt, worker


def _print_director_findings(director: ResearchDirector) -> None:
    if director.last_analysis:
        console.print("[bold grey19]Director analyzed the data and errors[/bold grey19]")
        for record in director.last_analysis:
            console.print(f"  {record.id}: {record.signature()}")
    if director.last_skill_selection.selected:
        console.print("[bold grey19]Director consulted skills[/bold grey19]")
        for selected in director.last_skill_selection.selected:
            console.print(f"  {selected.name}: {selected.reason}")


def _apply_plan_overrides(config: TaskConfig, overrides: dict) -> None:
    """Honor the safe frontmatter edits mid-run (goal, guardrails, timeouts)."""
    if overrides.get("goal"):
        config.goal = str(overrides["goal"])
    if overrides.get("guardrails"):
        config.guardrails = [str(item) for item in overrides["guardrails"]]
    if overrides.get("timeout_s"):
        config.agents.timeout_s = max(60, int(overrides["timeout_s"]))
    if overrides.get("budget_s"):
        config.agents.budget_s = max(60, int(overrides["budget_s"]))


async def _propose_round(
    director: ResearchDirector,
    store: RunStore,
    config: TaskConfig,
    session_dir: Path,
    round_number: int,
    notes: str,
    count: int,
    review: ReviewCallback | None,
) -> tuple[list[Idea], Path]:
    """Ask the director for the next round and write it as an editable plan file
    in the session folder.

    With a reviewer, loop until they approve (the possibly hand-edited file is
    what runs) or stop. Returns ([], path) when the reviewer ends the run."""
    plan_file = plan_path(session_dir, round_number)
    feedback: str | None = None
    previous: list[Idea] | None = None
    while True:
        with Activity(console, f"Director designing round {round_number}"):
            ideas = await director.propose(
                count=count if review is None else None,
                count_range=None if review is None else (PROPOSALS_MIN, PROPOSALS_MAX),
                round_number=round_number,
                attempts=store.load_attempts(),
                notes=notes,
                store=store,
                feedback=feedback,
                previous=previous,
            )
        _print_director_findings(director)
        plan_file.write_text(
            render_plan(
                round_number=round_number,
                ideas=ideas,
                goal=config.goal,
                metric=config.metric.name,
                guardrails=config.guardrails,
                timeout_s=config.agents.timeout_s,
                budget_s=config.agents.budget_s,
                analysis=list(director.last_analysis),
            )
        )
        refresh_session_readme(session_dir)
        if review is None:
            mark_executed(plan_file)
            return ideas, plan_file
        open_in_editor(plan_file)
        while True:
            decision = await asyncio.to_thread(review, round_number, ideas, plan_file)
            if decision.action == "stop":
                return [], plan_file
            if decision.action == "revise":
                feedback, previous = decision.feedback, ideas
                break
            # Approve: the hand-edited plan file is the source of truth.
            try:
                parsed = parse_plan(plan_file)
            except PlanError as exc:
                console.print(
                    f"[dark_orange3]! {exc}[/dark_orange3]\n"
                    "Fix the plan file and approve again, or give feedback / stop."
                )
                continue
            _apply_plan_overrides(config, parsed.overrides)
            mark_executed(plan_file)
            return parsed.ideas, plan_file


class RoundController:
    """Owns one round's agent tasks so the dashboard and the stop flag can
    cancel them safely from any thread."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        statuses: list[AgentStatus],
        tasks: list[asyncio.Task],
    ) -> None:
        self.loop = loop
        self.statuses = statuses
        self.tasks = tasks
        self.round_cancelled = False

    def cancel_agent(self, index: int) -> None:
        """Thread-safe: called from the dashboard's key listener."""
        self.loop.call_soon_threadsafe(self._cancel_agent, index)

    def cancel_round(self) -> None:
        """Thread-safe: cancel every agent and end the research run."""
        self.round_cancelled = True
        self.loop.call_soon_threadsafe(self._cancel_all)

    def _cancel_agent(self, index: int) -> None:
        status, task = self.statuses[index], self.tasks[index]
        if status.cancel_requested or task.done():
            return
        status.cancel_requested = True
        status.set(action="cancelling…")
        task.cancel()

    def _cancel_all(self) -> None:
        for index in range(len(self.tasks)):
            self._cancel_agent(index)


async def _watch_stop_flag(controller: RoundController, interval_s: float = 2.0) -> None:
    """Make `autoresearch stop` take effect mid-round instead of after it."""
    while True:
        await asyncio.sleep(interval_s)
        if Path(".autoresearch-stop").exists():
            controller.cancel_round()
            return


async def _run_round(
    config: TaskConfig,
    store: RunStore,
    repo: Path,
    ideas: list[Idea],
    round_number: int,
    base_ref: str,
    baseline: dict[str, float],
    header: str,
    stop_poll_s: float = 2.0,
) -> tuple[list[tuple[Attempt, WorkerResult | None]], bool]:
    """Run one round of agents under the live dashboard.

    Returns the per-agent results plus whether the whole round was cancelled
    (dashboard ``q`` or the stop flag)."""
    statuses = [
        AgentStatus(index=position + 1, attempt_id=uuid.uuid4().hex[:8], title=idea.title)
        for position, idea in enumerate(ideas)
    ]
    tasks = [
        asyncio.create_task(
            _execute_attempt(
                config,
                store,
                repo,
                idea,
                status.attempt_id,
                round_number,
                base_ref,
                baseline,
                status=status,
            ),
            name=f"agent-{status.attempt_id}",
        )
        for idea, status in zip(ideas, statuses, strict=True)
    ]
    controller = RoundController(asyncio.get_running_loop(), statuses, tasks)
    watcher = asyncio.create_task(_watch_stop_flag(controller, stop_poll_s), name="stop-watcher")

    # Lazy so a missing OPENROUTER_API_KEY (or any client failure) degrades to a
    # message instead of blocking the round. Runs on the dashboard's key thread.
    chat_holder: list[RoundChat] = []

    def _on_chat() -> None:
        if not chat_holder:
            try:
                chat_holder.append(
                    RoundChat(
                        openrouter_client(),
                        config.director.model,
                        config.director.temperature,
                        header=header,
                        ideas=ideas,
                        statuses=statuses,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - chat must never kill the round
                console.print(f"[red]round chat unavailable: {exc}[/red]")
                return
        chat_holder[0].session(console)

    try:
        with AgentDashboard(
            console,
            header,
            statuses,
            on_cancel_agent=controller.cancel_agent,
            on_cancel_round=controller.cancel_round,
            on_chat=_on_chat,
        ):
            raw = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
    failures = [
        item
        for item in raw
        if isinstance(item, BaseException) and not isinstance(item, asyncio.CancelledError)
    ]
    if failures:
        raise failures[0]
    results = [item for item in raw if isinstance(item, tuple)]
    return results, controller.round_cancelled


async def run_research(
    config: TaskConfig,
    *,
    goal: str | None = None,
    guardrails: list[str] | None = None,
    parallel: int | None = None,
    max_experiments: int | None = None,
    max_cost: float | None = None,
    resume_store: RunStore | None = None,
    initial_ideas: list[Idea] | None = None,
    initial_plan_path: Path | None = None,
    session_dir: Path | None = None,
    review: ReviewCallback | None = None,
) -> RunStore:
    if goal:
        config.goal = goal
    if guardrails:
        config.guardrails.extend(guardrails)
    parallel = parallel or config.agents.count
    if max_experiments:
        maximum = max_experiments
    elif review is not None:
        # The human reviewer is the budget; cap only at the theoretical maximum.
        maximum = config.budget.rounds * PROPOSALS_MAX
    else:
        maximum = config.budget.max_experiments
    seed_template = config.resolve(config.workspace.seed)
    if not seed_template.exists():
        raise RuntimeError(f"seed workspace does not exist: {seed_template}; run prepare.py first")
    main_ref = _main_ref(config)
    if resume_store:
        store = resume_store
        repo = store.run_dir / "repo"
        if not (repo / ".git").exists():
            raise RuntimeError(f"run repository is missing: {repo}")
        prior = store.load_state()
        baseline = prior["baseline"]
        incumbent = prior["incumbent"]
        incumbent_ref = prior.get("incumbent_ref", main_ref)
        completed = int(prior.get("completed", len(store.load_attempts())))
        round_number = int(prior.get("round", 0))
        # Continue the same session folder so the notebook stays in one place.
        if session_dir is None and prior.get("session_dir"):
            session_dir = Path(prior["session_dir"])
    else:
        runs_root = config.resolve(config.workspace.runs)
        store = RunStore.create(runs_root, config.name)
        shutil.copy2(config.config_path, store.run_dir / "task.yaml")
        repo = store.run_dir / "repo"
        shutil.copytree(seed_template, repo, ignore=_ignore_runs_root(runs_root))
        await ensure_seed_repo(repo)
        with Activity(console, "Confirming the baseline before research"):
            baseline_eval = await validate_baseline(config)
        if baseline_eval.error:
            raise RuntimeError(f"baseline evaluation failed: {baseline_eval.error}")
        if baseline_eval.validation_frame is not None:
            store.save_validation_frame("baseline", baseline_eval.validation_frame)
        baseline = dict(baseline_eval.metrics)
        holdout_key = f"holdout_{config.metric.name}"
        baseline[holdout_key] = baseline_eval.holdout_metrics[config.metric.name]
        incumbent = baseline
        incumbent_ref = main_ref
        completed = 0
        round_number = 0
    if session_dir is None:
        session_dir = new_session_dir(plans_dir_for_task(config.root), config.goal)
    if not resume_store:
        store.save_state(
            {
                "task": config.name,
                "status": "running",
                "goal": config.goal,
                "primary_metric": config.metric.name,
                "metric_direction": config.metric.direction,
                "baseline": baseline,
                "incumbent": baseline,
                "incumbent_ref": main_ref,
                "completed": 0,
                "round": 0,
                "session_dir": str(session_dir),
            }
        )
    console.print(
        f"[bold grey19]Baseline[/bold grey19] {config.metric.name}={baseline[config.metric.name]:.6f}, "
        f"holdout={baseline[f'holdout_{config.metric.name}']:.6f}"
    )
    director = ResearchDirector(config)
    stop_requested = False

    def total_spend() -> float:
        return sum(a.metadata.get("cost_usd", 0.0) or 0.0 for a in store.load_attempts())

    try:
        while completed < maximum and round_number < config.budget.rounds:
            if Path(".autoresearch-stop").exists():
                stop_requested = True
                console.print(
                    "[dark_orange3]Stop flag found (.autoresearch-stop) - stopping the run "
                    "before the next round.[/dark_orange3]"
                )
                break
            spent = total_spend()
            if max_cost is not None and spent >= max_cost:
                stop_requested = True
                console.print(
                    f"[bold grey19]Cost cap reached[/bold grey19]: spent {format_cost(spent)} of "
                    f"{format_cost(max_cost)} budget - stopping before the next round."
                )
                break
            round_number += 1
            count = min(parallel, maximum - completed)
            notes = store.notes_file.read_text() if store.notes_file.exists() else ""
            # Pre-seeded opening-round ideas (from the interview, or pinned in the
            # task config) skip director generation entirely for round 1.
            seeded = initial_ideas
            from_config = False
            if round_number == 1 and not seeded and config.opening_round.ideas:
                seeded = config.opening_round.ideas
                from_config = True
            if round_number == 1 and seeded:
                if from_config and config.opening_round.think_seconds > 0:
                    # Cosmetic pause so the director's 'thinking' shows briefly
                    # before the pre-seeded plan appears.
                    with Activity(console, f"Director designing round {round_number}"):
                        await asyncio.sleep(config.opening_round.think_seconds)
                if from_config:
                    console.print(
                        f"[bold grey19]Opening round[/bold grey19]: using "
                        f"{len(config.opening_round.ideas)} pre-seeded ideas from the task config"
                    )
                ideas = seeded[:count]
                plan_file = initial_plan_path
            else:
                ideas, plan_file = await _propose_round(
                    director, store, config, session_dir, round_number, notes, count, review
                )
                if not ideas:
                    console.print("[bold grey19]Run ended at review.[/bold grey19]")
                    break
                ideas = ideas[: maximum - completed]
            count = len(ideas)
            console.print(
                f"[bold blue]Round {round_number}[/bold blue]: launching {count} experiments"
            )
            header = (
                f"Round {round_number} · baseline {config.metric.name} "
                f"{incumbent[config.metric.name]:.4f} · {count} agents"
            )
            results, round_cancelled = await _run_round(
                config, store, repo, ideas, round_number, incumbent_ref, incumbent, header
            )
            # Cancelled attempts should not burn the experiment budget.
            completed += sum(1 for attempt, _ in results if attempt.status != "cancelled")
            passing = [
                (attempt, worker)
                for attempt, worker in results
                if attempt.status == "passed"
                and worker is not None
                and config.metric.name in attempt.metrics
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
                    incumbent_ref = main_ref
                    incumbent = dict(attempt.metrics)
                    incumbent[f"holdout_{config.metric.name}"] = attempt.holdout_metrics[
                        config.metric.name
                    ]
                    console.print(
                        f"[green]Promoted {attempt.id}[/green]: "
                        f"{config.metric.name}={attempt.metrics[config.metric.name]:.6f}"
                    )
                    if config.runtime == "foundry":
                        # The winner goes live: its code becomes the published
                        # transform on the main Foundry branch.
                        code, output = await _run(
                            "git", "push", "origin", f"HEAD:{main_ref}", cwd=repo
                        )
                        if code:
                            console.print(
                                f"[dark_orange3]! could not push the promoted model to Foundry "
                                f"{main_ref}: {output.strip()}[/dark_orange3]"
                            )
                        else:
                            console.print(
                                f"[green]Pushed the promoted model to Foundry {main_ref}[/green]"
                            )
                store.save_attempt(attempt)
            if round_cancelled:
                # Skip the director's reflection: every agent was interrupted.
                reflection = "Round cancelled by the user before completion."
            else:
                with Activity(console, f"Director reviewing round {round_number} results"):
                    reflection = await director.reflect([attempt for attempt, _ in results])
            store.append_note(f"Round {round_number}", reflection)
            if plan_file is not None:
                findings_file = write_findings(
                    plan_file,
                    [attempt for attempt, _ in results],
                    config.metric.name,
                    reflection,
                )
                if findings_file is not None and review is not None and not round_cancelled:
                    console.print(f"[dim]Findings written to {findings_file}[/dim]")
                    open_in_editor(findings_file)
            for _, worker in results:
                # Agents cancelled mid-build (worker is None) already removed theirs.
                if worker is not None:
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
                    "cost_usd": round(total_spend(), 6),
                    "session_dir": str(session_dir),
                }
            )
            if round_cancelled:
                stop_requested = True
                console.print("[bold grey19]Round cancelled - research run stopped.[/bold grey19]")
                break
    finally:
        state = store.load_state()
        state["status"] = (
            "stopped"
            if stop_requested or Path(".autoresearch-stop").exists()
            else "completed"
        )
        state["cost_usd"] = round(total_spend(), 6)
        store.save_state(state)
        Path(".autoresearch-stop").unlink(missing_ok=True)
    return store
