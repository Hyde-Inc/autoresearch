"""Trigger training on Foundry: push the transforms repo, then build a dataset.

This is the ``TRAIN`` step of the autoresearch loop when a task runs with
``runtime: foundry``. The model code lives in a Foundry Python transforms repo
that is git-backed by Foundry's Stemma remote. We:

1. commit the agent's edits and ``git push`` (Foundry publishes the transform),
2. create a build of the output dataset (``POST /api/v2/orchestration/builds/create``),
3. poll the build until it reaches a terminal state.

Credentials come from the environment via :mod:`autoresearch.foundry`
(``FOUNDRY_HOSTNAME`` / ``FOUNDRY_TOKEN``; the token needs
``api:orchestration-write``). The git remote already carries its own auth.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import foundry

_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED", "ABORTED"}


class FoundryBuildError(RuntimeError):
    """A push or build step failed in a way the user can act on."""


@dataclass
class BuildResult:
    build_rid: str
    status: str
    job_rids: list[str]

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"


def _git(repo_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise FoundryBuildError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def push_repo(repo_dir: str | Path, *, branch: str = "master", message: str) -> str | None:
    """Commit any changes in ``repo_dir`` and push to Foundry. Returns commit sha.

    A no-op (nothing to commit) returns ``None`` but still pushes, so a build can
    reuse the already-published head.
    """
    repo = Path(repo_dir)
    if not (repo / ".git").exists():
        raise FoundryBuildError(f"{repo} is not a git repository")

    _git(repo, "add", "-A")
    status = _git(repo, "status", "--porcelain")
    committed: str | None = None
    if status.stdout.strip():
        _git(repo, "commit", "-m", message)
        committed = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Foundry can advance the remote branch (server-side commits, other clients),
    # so integrate it before pushing or the push is rejected as non-fast-forward.
    fetch = _git(repo, "fetch", "origin", branch, check=False)
    if fetch.returncode == 0:
        rebase = _git(repo, "rebase", f"origin/{branch}", check=False)
        if rebase.returncode != 0:
            _git(repo, "rebase", "--abort", check=False)
            raise FoundryBuildError(
                f"could not rebase local changes onto Foundry origin/{branch}:\n"
                f"{rebase.stderr.strip() or rebase.stdout.strip()}\n"
                "Resolve the conflict in the transforms repo, then retry."
            )

    push = _git(repo, "push", "origin", f"HEAD:{branch}", check=False)
    if push.returncode != 0:
        raise FoundryBuildError(
            "git push to Foundry failed (transform will not publish):\n"
            f"{push.stderr.strip() or push.stdout.strip()}"
        )
    return committed


def create_build(
    target_rids: list[str],
    *,
    branch: str = "master",
    retry_count: int = 0,
    force: bool = False,
    fallback_branches: list[str] | None = None,
) -> str:
    """Trigger a manual build of ``target_rids`` and return the build RID.

    ``force`` sets ``forceBuild`` to rebuild even when Foundry considers the
    output up to date (staleness ignored). ``fallback_branches`` lets a build on
    an experiment branch resolve input datasets that only have transactions on
    the main branch.
    """
    payload = foundry._api_json(
        "POST",
        "/api/v2/orchestration/builds/create",
        body={
            "abortOnFailure": False,
            "forceBuild": force,
            "retryBackoffDuration": {"unit": "SECONDS", "value": 30},
            "retryCount": retry_count,
            "fallbackBranches": list(fallback_branches or []),
            "branchName": branch,
            "target": {"type": "manual", "targetRids": target_rids},
        },
    )
    build_rid = payload.get("rid")
    if not build_rid:
        raise FoundryBuildError(f"build create returned no RID: {payload}")
    return str(build_rid)


# Both mean "the code we just pushed hasn't republished yet": no job spec at all
# (first publish) or the old spec still matches the output (new publish pending).
_PUBLISH_PENDING = ("BuildTargetsMissingJobSpecs", "BuildTargetsUpToDate")


def create_build_when_ready(
    target_rids: list[str],
    *,
    branch: str = "master",
    force: bool = False,
    publish_timeout_s: int = 1500,
    poll_s: int = 20,
    on_wait=None,
) -> str:
    """Create a build, retrying while the just-pushed transform is still publishing.

    After a push, Foundry runs a publish/CI job before the new logic takes effect;
    until then ``create_build`` returns ``BuildTargetsMissingJobSpecs`` (first
    publish) or ``BuildTargetsUpToDate`` (new version not live yet). We poll until
    a build is warranted or ``publish_timeout_s`` elapses. ``on_wait`` is an
    optional ``(elapsed_s: float)`` progress callback.
    """
    start = time.monotonic()
    while True:
        try:
            return create_build(target_rids, branch=branch, force=force)
        except foundry.FoundryError as exc:
            if not any(name in str(exc) for name in _PUBLISH_PENDING):
                raise
            elapsed = time.monotonic() - start
            if elapsed > publish_timeout_s:
                raise FoundryBuildError(
                    "the pushed transform never became buildable within "
                    f"{publish_timeout_s}s (still up-to-date / no job spec). "
                    "Open the repository's Checks tab in Foundry — a failing "
                    "publish/CI job, or an unchanged transform, would explain this. "
                    "Use --force to rebuild unchanged code."
                ) from exc
            if on_wait:
                on_wait(elapsed)
            time.sleep(poll_s)


def get_build(build_rid: str) -> dict:
    return foundry._api_json("GET", f"/api/v2/orchestration/builds/{build_rid}")


def cancel_build(build_rid: str) -> None:
    """Ask Foundry to cancel a running build (all of its unfinished jobs)."""
    foundry._api_json("POST", f"/api/v2/orchestration/builds/{build_rid}/cancel")


def _job_diagnostics(job_rids: list[str]) -> str:
    messages: list[str] = []
    for job_rid in job_rids[:4]:
        try:
            job = foundry._api_json("GET", f"/api/v2/orchestration/jobs/{job_rid}")
        except foundry.FoundryError:
            continue
        detail = job.get("errorMessage") or job.get("statusReason") or ""
        if detail:
            messages.append(f"{job_rid}: {detail}")
    return "\n".join(messages)


def wait_for_build(
    build_rid: str,
    *,
    timeout_s: int = 900,
    poll_s: int = 15,
    on_status=None,
) -> BuildResult:
    """Poll a build until terminal. Raises :class:`FoundryBuildError` on failure.

    A build that outlives ``timeout_s`` is actively cancelled on Foundry (so it
    stops consuming compute) before the error is raised. ``on_status`` is an
    optional callback ``(status: str, elapsed_s: float)`` for surfacing progress
    to the dashboard.
    """
    start = time.monotonic()
    last_status = ""
    while True:
        build = get_build(build_rid)
        status = str(build.get("status", "")).upper()
        job_rids = [str(j) for j in build.get("jobRids", [])]
        elapsed = time.monotonic() - start
        if on_status and status != last_status:
            on_status(status, elapsed)
            last_status = status
        if status in _TERMINAL:
            result = BuildResult(build_rid=build_rid, status=status, job_rids=job_rids)
            if not result.succeeded:
                diagnostics = _job_diagnostics(job_rids)
                raise FoundryBuildError(
                    f"Foundry build {build_rid} ended {status}."
                    + (f"\n{diagnostics}" if diagnostics else "")
                )
            return result
        if elapsed > timeout_s:
            cancelled = "cancelled on Foundry"
            try:
                cancel_build(build_rid)
            except foundry.FoundryError as exc:
                cancelled = f"cancel request failed: {exc}"
            raise FoundryBuildError(
                f"Foundry build {build_rid} exceeded the {timeout_s}s timeout "
                f"(last status {status or 'unknown'}); {cancelled}"
            )
        time.sleep(poll_s)
