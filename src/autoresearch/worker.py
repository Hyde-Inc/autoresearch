from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TaskConfig
from .models import Idea
from .opencode import OpenCodeResult, run_opencode
from .store import RunStore


@dataclass
class WorkerResult:
    branch: str
    worktree: Path
    commit: str | None
    diff: str
    changed_paths: list[str]
    opencode: OpenCodeResult
    error: str | None = None


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


def build_worker_prompt(workspace: Path, task_contract: str, experiment: str) -> str:
    return (
        f"You are an autonomous ML research engineer assigned to exactly this workspace: "
        f"{workspace}.\n\n"
        f"TASK CONTRACT\n{task_contract}\n\n"
        f"ASSIGNED EXPERIMENT\n{experiment}\n\n"
        "This is all the setup context you need. Start by inspecting solution/train.py, then "
        "implement only the assigned hypothesis. Do not search for TASK.md, EXPERIMENT.md, or "
        "other repository context. "
        f"Read and write only inside {workspace}; never inspect the parent directory, sibling "
        "worktrees, the source seed, or other experiments. Only modify files under solution/. "
        "Do not modify pyproject.toml, install system packages, use subagents, or create todo lists. "
        "Run focused local checks from this workspace using the provided training data. "
        "The protected evaluator will run solution/train.py and compare forecasts.parquet. "
        "Keep runtime within the stated budget. Do not ask questions or commit changes."
    )


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
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> WorkerResult:
    branch = f"autoresearch/{attempt_id}-{_slug(idea.title)}"
    worktree = store.worktrees_dir / attempt_id
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
    (worktree / "EXPERIMENT.md").write_text(experiment)
    workspace = worktree.resolve()
    task_contract = (worktree / "TASK.md").read_text()
    prompt = build_worker_prompt(workspace, task_contract, experiment)
    log_path = store.logs_dir / f"{attempt_id}.jsonl"
    result = await run_opencode(
        worktree,
        prompt,
        config.agents.model,
        config.agents.timeout_s,
        log_path,
        on_event=on_event,
    )
    (worktree / "EXPERIMENT.md").unlink(missing_ok=True)
    _, changed = await _run("git", "status", "--porcelain", cwd=worktree)
    changed_paths = [line[3:] for line in changed.splitlines() if len(line) > 3]
    _, diff = await _run("git", "diff", "--binary", cwd=worktree)
    commit: str | None = None
    error = result.error
    if result.returncode != 0 and error is None:
        error = f"opencode exited with status {result.returncode}"
    if not error and changed_paths:
        await _run("git", "add", "solution", cwd=worktree)
        code, output = await _run(
            "git", "commit", "-m", f"experiment: {idea.title}", cwd=worktree
        )
        if code:
            error = output
        else:
            _, commit = await _run("git", "rev-parse", "HEAD", cwd=worktree)
            commit = commit.strip()
            _, diff = await _run("git", "show", "--format=", "--binary", "HEAD", cwd=worktree)
    elif not changed_paths and not error:
        error = "agent made no changes"
    return WorkerResult(branch, worktree, commit, diff, changed_paths, result, error)


async def remove_worktree(repo: Path, result: WorkerResult) -> None:
    await _run("git", "worktree", "remove", "--force", str(result.worktree), cwd=repo)
