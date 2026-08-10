from __future__ import annotations

import ast
import asyncio
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .agent_status import COMMITTING, EVALUATING, EXPLORING, SETTING_UP, TRAINING
from .config import TaskConfig
from .harness import evaluate_validation
from .metrics import load_task_spec
from .models import Idea
from .opencode import OpenCodeResult, run_opencode
from .skills import load_skills, named_skill_context
from .store import RunStore

MAX_SESSIONS = 10
"""Hard cap on opencode sessions per experiment; the budget usually bites first."""
MIN_FOLLOWUP_S = 120
"""Don't start a fix/improve session with less budget remaining than this."""
STALL_LIMIT = 2
"""Stop after this many consecutive scored sessions without a new best metric."""


@dataclass
class WorkerResult:
    branch: str
    worktree: Path
    commit: str | None
    diff: str
    changed_paths: list[str]
    opencode: OpenCodeResult
    error: str | None = None
    sessions: int = 0
    history: list[dict] = field(default_factory=list)
    """One entry per session: which step it reached and how it ended."""


async def _run(*args: str, cwd: Path) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    return process.returncode or 0, output.decode(errors="replace")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:36] or "experiment"


def _better(candidate: float, incumbent: float, direction: str) -> bool:
    return candidate < incumbent if direction == "min" else candidate > incumbent


_PROMPT_RULES = (
    "You may only modify files under solution/. Do not run full training or wait on long "
    "processes yourself - the orchestrator trains and evaluates for free after your reply "
    "ends, and every minute you spend polling burns paid session budget; limit local checks "
    "to a quick smoke test on a small slice. Do not ask questions. Do not commit changes."
)


def _fix_prompt(failure: str) -> str:
    return (
        "The orchestrator ran your solution and it did not produce a valid scored result:\n\n"
        f"{failure[:4000]}\n\n"
        "Diagnose and fix the problem so solution/train.py runs end to end within the runtime "
        f"budget. Re-read TASK.md and EXPERIMENT.md if you need to. {_PROMPT_RULES}"
    )


def _improve_prompt(metric: str, value: float, target: float | None, direction: str) -> str:
    goal = "lower" if direction == "min" else "higher"
    comparison = (
        f" It must be {goal} than the incumbent's {metric}={target:.6f} to be promoted."
        if target is not None
        else ""
    )
    return (
        "The orchestrator trained and scored your solution on the validation split: "
        f"{metric}={value:.6f}.{comparison} Improve the solution - prefer one focused change "
        "over a rewrite, and keep solution/train.py runnable end to end at all times. "
        f"{_PROMPT_RULES}"
    )


def _verify_build(worktree: Path) -> str | None:
    """Cheap build check before spending a training run: the entry point must
    exist and parse."""
    train = worktree / "solution" / "train.py"
    if not train.exists():
        return "solution/train.py does not exist"
    try:
        ast.parse(train.read_text(), filename="solution/train.py")
    except SyntaxError as exc:
        return f"solution/train.py has a syntax error: {exc}"
    return None


async def _snapshot(worktree: Path, title: str, session: int) -> tuple[str | None, str | None]:
    """Commit the current solution/ state after one session.

    Returns ``(sha, error)``; ``sha`` is None when nothing changed since the
    previous snapshot."""
    await _run("git", "add", "solution", cwd=worktree)
    staged, _ = await _run("git", "diff", "--cached", "--quiet", cwd=worktree)
    if staged == 0:
        return None, None
    code, output = await _run(
        "git", "commit", "-m", f"experiment: {title} (session {session})", cwd=worktree
    )
    if code:
        return None, output
    _, sha = await _run("git", "rev-parse", "HEAD", cwd=worktree)
    return sha.strip(), None


async def ensure_seed_repo(seed: Path) -> None:
    if (seed / ".git").exists():
        return
    code, output = await _run("git", "init", "-b", "main", cwd=seed)
    if code:
        raise RuntimeError(output)
    await _run("git", "config", "user.name", "Autoresearch", cwd=seed)
    await _run("git", "config", "user.email", "autoresearch@localhost", cwd=seed)
    await _run("git", "add", ".", cwd=seed)
    code, output = await _run("git", "commit", "-m", "Seed experiment", cwd=seed)
    if code:
        raise RuntimeError(output)


async def run_worker(
    *,
    config: TaskConfig,
    store: RunStore,
    repo: Path,
    idea: Idea,
    attempt_id: str,
    base_ref: str,
    baseline: dict[str, float] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_phase: Callable[[str, str], None] | None = None,
) -> WorkerResult:
    """Run one experiment as a build -> train -> evaluate loop with a coding agent.

    Instead of a single monolithic session, the worker converses with one
    opencode session the way a human drives a coding assistant, and each step
    is verified and surfaced on the dashboard before the next one runs:

    1. BUILD - the agent writes solution/train.py (paid session). The step is
       verified cheaply (the file exists and parses) before any training run.
    2. TRAIN - the orchestrator itself runs solution/train.py on the
       validation request. No opencode process is alive here, so no tokens are
       burned waiting on training; train.py's output streams to the dashboard.
    3. EVALUATE - the harness verifies the forecasts (coverage, finiteness)
       and scores them; the outcome (traceback, guardrail failure, or metric
       vs. the incumbent) goes back into the same session to fix or improve.

    Each session's solution/ state is snapshot-committed, so a timeout or a
    late regression never loses earlier working states - the branch is reset
    to the best-scoring snapshot at the end. Per-session step outcomes are
    returned in ``WorkerResult.history``. The loop stops when a clean score
    beats the incumbent, when the experiment budget (``agents.budget_s``) or
    ``MAX_SESSIONS`` runs out, when the same failure repeats twice, or after
    ``STALL_LIMIT`` scored sessions without a new best. Holdout metrics are
    never fed back, so the agent cannot iterate against the final gate.
    """

    def phase(name: str, action: str) -> None:
        if on_phase is not None:
            on_phase(name, action)

    branch = f"autoresearch/{attempt_id}-{_slug(idea.title)}"
    worktree = store.worktrees_dir / attempt_id
    phase(SETTING_UP, "creating git worktree")
    if worktree.exists():
        shutil.rmtree(worktree)
    code, output = await _run(
        "git", "worktree", "add", "-b", branch, str(worktree), base_ref, cwd=repo
    )
    if code:
        return WorkerResult(branch, worktree, None, "", [], OpenCodeResult(code, None, output), output)

    experiment = (
        f"# Experiment: {idea.title}\n\n"
        f"## Hypothesis\n{idea.hypothesis}\n\n"
        f"## Instructions\n{idea.instructions}\n"
    )
    if idea.skills_used:
        skill_context = named_skill_context(idea.skills_used, load_skills(config))
        experiment += f"\n## Research skills used\n{', '.join(idea.skills_used)}\n"
        if skill_context:
            experiment += (
                "\n## Implementation playbooks\n\n"
                "Apply the relevant guidance below to this experiment. Verify assumptions "
                "against the installed package versions and task contract.\n\n"
                f"{skill_context}\n"
            )
    if config.context:
        experiment += f"\n## Business context\n{config.context}\n"
    spec = load_task_spec(config)
    if spec is not None:
        experiment += (
            f"\n## Evaluation metric: {spec.name} ({spec.direction}imize)\n"
            f"{spec.understanding}\n\n"
            "The protected evaluator scores forecasts with exactly this code:\n\n"
            f"```python\n{spec.code}\n```\n"
        )
    (worktree / "EXPERIMENT.md").write_text(experiment)
    prompt = (
        "You are an autonomous ML research engineer. Read TASK.md and EXPERIMENT.md. "
        "Implement this experiment completely. Division of labor: you BUILD solution/train.py; "
        "after your reply ends, the orchestrator TRAINS it on the real evaluation inputs and "
        "SCORES the forecasts, then reports the result back into this conversation so you can "
        "fix or improve it. Get a complete end-to-end solution/train.py written before "
        "polishing details, and keep its runtime within the stated budget. "
        f"{_PROMPT_RULES}"
    )
    log_path = store.logs_dir / f"{attempt_id}.jsonl"
    # Session state must outlive individual opencode runs so the loop can
    # resume the same conversation with evaluator feedback.
    scratch = store.run_dir / "opencode" / attempt_id
    primary = config.metric.name
    direction = config.metric.direction
    target = (baseline or {}).get(primary)
    deadline = time.monotonic() + config.agents.budget_s
    session_id: str | None = None
    last_result = OpenCodeResult(0, None, None)
    sessions = 0
    head: str | None = None  # newest snapshot commit
    best_commit: str | None = None  # snapshot with the best clean validation score
    best_value: float | None = None
    stall = 0
    last_failure: str | None = None
    error: str | None = None
    history: list[dict] = []

    def record(step: str, outcome: str, commit: str | None, metric: float | None = None) -> None:
        history.append(
            {
                "session": sessions,
                "step": step,
                "outcome": outcome[:300],
                "commit": commit[:8] if commit else None,
                "metric": metric,
            }
        )

    def training_progress(line: str) -> None:
        phase(TRAINING, line)

    try:
        while sessions < MAX_SESSIONS:
            remaining = deadline - time.monotonic()
            # The first session always runs; follow-ups need enough budget left
            # to plausibly change the outcome.
            if remaining <= (0 if sessions == 0 else MIN_FOLLOWUP_S):
                break
            sessions += 1
            phase(
                EXPLORING,
                "coding agent building the solution"
                if sessions == 1
                else f"session {sessions}: applying evaluator feedback",
            )
            last_result = await run_opencode(
                worktree,
                prompt,
                config.agents.model,
                max(1, min(config.agents.timeout_s, int(remaining))),
                log_path,
                on_event=on_event,
                session_id=session_id,
                data_dir=scratch,
            )
            session_id = last_result.session_id or session_id
            timed_out = last_result.error is not None and "timeout" in last_result.error
            note = "session timed out mid-build; " if timed_out else ""
            phase(COMMITTING, f"session {sessions}: snapshotting solution/")
            sha, commit_error = await _snapshot(worktree, idea.title, sessions)
            if commit_error:
                error = commit_error
                record("commit", commit_error, head)
                break
            if sha is None:
                if head is None:
                    # Nothing to harvest at all: surface the agent's own failure.
                    error = last_result.error or "agent made no changes"
                record("build", note + "no changes to solution/", head)
                break  # a session that changed nothing cannot make progress
            head = sha
            if last_result.error and not timed_out:
                record("build", last_result.error, head)
                break  # hard CLI/provider failure; keep what was harvested
            # Verify the build before spending a training run on it.
            build_error = _verify_build(worktree)
            if build_error is not None:
                check, value, failure = None, None, build_error
                phase(EVALUATING, f"session {sessions}: build check failed")
                record("build", note + build_error, head)
            else:
                phase(TRAINING, f"session {sessions}: training on the validation split")
                check = await evaluate_validation(
                    worktree,
                    config,
                    baseline=baseline,
                    guardrails=config.guardrails,
                    on_progress=training_progress,
                )
                (worktree / config.data.output).unlink(missing_ok=True)
                value = check.metrics.get(primary)
                failure = (
                    check.error
                    or "; ".join(check.guardrail_failures)
                    or (f"{primary} missing from evaluation metrics" if value is None else None)
                )
            if failure is None:
                last_failure = None
                if best_value is None or _better(value, best_value, direction):
                    best_value, best_commit, stall = value, head, 0
                else:
                    stall += 1
                summary = f"{primary} {value:.6f}" + (
                    f" vs incumbent {target:.6f}" if target is not None else ""
                )
                phase(EVALUATING, f"session {sessions}: {summary}")
                record("scored", note + summary, head, value)
                if target is None or _better(value, target, direction):
                    break  # done: a clean score that beats the incumbent
                if stall >= STALL_LIMIT:
                    break
                prompt = _improve_prompt(primary, value, target, direction)
            else:
                if build_error is None:
                    step = "train" if failure.startswith(("training exceeded", "solution exited")) else "evaluate"
                    phase(EVALUATING, f"session {sessions}: {failure}")
                    record(step, note + failure, head)
                if failure == last_failure:
                    break  # identical failure twice in a row: the agent is stuck
                last_failure = failure
                prompt = _fix_prompt(failure)
    finally:
        # The scratch XDG root holds a copy of opencode's auth.json.
        shutil.rmtree(scratch, ignore_errors=True)
    if best_commit is not None and best_commit != head:
        # A later session regressed a previously scored state; promote the best.
        phase(COMMITTING, "restoring best-scoring snapshot")
        await _run("git", "reset", "--hard", best_commit, cwd=worktree)
        head = best_commit
    (worktree / "EXPERIMENT.md").unlink(missing_ok=True)
    (worktree / config.data.output).unlink(missing_ok=True)
    phase(COMMITTING, "reviewing changes")
    _, porcelain = await _run("git", "status", "--porcelain", cwd=worktree)
    changed_paths = {line[3:] for line in porcelain.splitlines() if len(line) > 3}
    commit = head
    if commit is not None:
        _, committed = await _run("git", "diff", "--name-only", base_ref, "HEAD", cwd=worktree)
        changed_paths.update(name.strip() for name in committed.splitlines() if name.strip())
        _, diff = await _run("git", "diff", "--binary", base_ref, "HEAD", cwd=worktree)
    else:
        _, diff = await _run("git", "diff", "--binary", cwd=worktree)
    if error is None and commit is None:
        error = "agent made no changes"
    return WorkerResult(
        branch,
        worktree,
        commit,
        diff,
        sorted(changed_paths),
        last_result,
        error=error,
        sessions=sessions,
        history=history,
    )


async def remove_worktree(repo: Path, result: WorkerResult) -> None:
    await remove_worktree_path(repo, result.worktree)


async def remove_worktree_path(repo: Path, worktree: Path) -> None:
    """Remove a worktree by path; safe even if git never registered it."""
    await _run("git", "worktree", "remove", "--force", str(worktree), cwd=repo)
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    await _run("git", "worktree", "prune", cwd=repo)
