from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import TaskConfig
from .metrics import load_task_spec
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
    if idea.skills_used:
        experiment += f"\n## Research skills used\n{', '.join(idea.skills_used)}\n"
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
        "Implement this experiment completely. You may only modify files under solution/. "
        "Run local checks using training data, but do not ask questions. The protected evaluator "
        "will run solution/train.py and compare forecasts.parquet. Keep runtime within the stated "
        "budget. Do not commit changes."
    )
    log_path = store.logs_dir / f"{attempt_id}.jsonl"
    result = await run_opencode(
        worktree, prompt, config.agents.model, config.agents.timeout_s, log_path
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
