"""Foundry runtime for the experiment loop: TRAIN happens as a Foundry build.

Mirrors :mod:`autoresearch.harness` (same :class:`Evaluation` contract) but
instead of running ``solution/train.py`` locally:

1. the experiment worktree's HEAD is pushed to a Foundry branch named after the
   experiment's git branch (force-push - experiment branches belong to us),
2. the ``forecasts`` dataset is built **on that branch** (input datasets fall
   back to the main branch), so parallel agents each get their own published
   transform and their own forecasts branch without colliding,
3. the forecasts and the sealed actuals are read back through readTable and
   scored locally with the exact same metrics and guardrails as the local
   harness.

One build serves both splits: the transform forecasts every requested
(sku, date) in ``forecast_request``, which covers validation and holdout, so
the final promotion gate only re-reads the already-built branch.

All Foundry/git calls are synchronous; the async entry points run them in a
worker thread so the dashboard keeps rendering while builds poll.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from . import foundry
from .config import TaskConfig
from .foundry_build import FoundryBuildError, create_build, wait_for_build
from .guardrails import check_guardrails
from .harness import Evaluation, forecasting_metrics
from .metrics import compile_metric, load_task_spec

OnProgress = Callable[[str], None]

PUBLISH_TIMEOUT_S = 1500
"""How long to wait for Foundry's publish/CI job after a push before giving up."""


def _git(worktree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=worktree, capture_output=True, text=True, check=False
    )


def _current_branch(worktree: Path) -> str | None:
    result = _git(worktree, "branch", "--show-current")
    return result.stdout.strip() or None


def _push_head(worktree: Path, branch: str, *, force: bool) -> bool:
    """Push the worktree's committed HEAD to ``origin/<branch>``.

    Only committed state is pushed (the worker snapshots the allowed paths
    before evaluating), so scratch files like EXPERIMENT.md never reach
    Foundry. Returns True when the remote branch actually changed - the signal
    that a republish (and therefore a rebuild) is expected.
    """
    head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    _git(worktree, "fetch", "origin", branch)  # may fail: branch may not exist yet
    remote = _git(worktree, "rev-parse", f"origin/{branch}").stdout.strip() or None
    if remote == head:
        return False
    args = ["push", "-f", "origin", f"HEAD:{branch}"] if force else [
        "push", "origin", f"HEAD:{branch}"
    ]
    result = _git(worktree, *args)
    if result.returncode != 0:
        raise FoundryBuildError(
            f"git push to Foundry branch {branch} failed:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return True


def _start_build(
    config: TaskConfig,
    branch: str,
    *,
    expect_new_code: bool,
    on_progress: OnProgress | None,
) -> str | None:
    """Trigger a build of the forecasts dataset on ``branch``.

    Returns the build RID, or ``None`` when Foundry reports the output is
    already up to date *and* we did not just push new code (baseline case /
    final gate re-reading the validation build). When new code was pushed,
    ``UpToDate`` and ``MissingJobSpecs`` both mean "publish still in flight",
    so we poll until the build is accepted.
    """
    fdry = config.foundry
    assert fdry is not None
    fallback = [fdry.branch] if branch != fdry.branch else []
    start = time.monotonic()
    while True:
        try:
            return create_build(
                [fdry.datasets.forecasts], branch=branch, fallback_branches=fallback
            )
        except foundry.FoundryError as exc:
            text = str(exc)
            if "BuildTargetsUpToDate" in text and not expect_new_code:
                return None
            if not any(
                name in text for name in ("BuildTargetsMissingJobSpecs", "BuildTargetsUpToDate")
            ):
                raise
            elapsed = time.monotonic() - start
            if elapsed > PUBLISH_TIMEOUT_S:
                raise FoundryBuildError(
                    f"the pushed transform on branch {branch} never became buildable "
                    f"within {PUBLISH_TIMEOUT_S}s - check the repository's Checks tab "
                    "in Foundry for a failing publish job"
                ) from exc
            if on_progress is not None:
                on_progress(f"waiting for Foundry to publish the transform ({int(elapsed)}s)")
            time.sleep(fdry.poll_s)


def _normalize_dates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = frame.copy()
    dates = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
    frame[column] = dates.dt.normalize()
    return frame


def _score_split(
    forecasts: pd.DataFrame, actuals: pd.DataFrame, config: TaskConfig
) -> tuple[dict[str, float], pd.DataFrame]:
    """Score forecasts against one split's sealed actuals.

    The forecasts frame covers the full request (validation + holdout), so it
    may contain more rows than this split - but it must cover every requested
    row of the split exactly once.
    """
    data = config.data
    required = [data.id_column, data.date_column, "forecast"]
    missing = set(required) - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts dataset is missing columns: {sorted(missing)}")
    keys = [data.id_column, data.date_column]
    forecasts = _normalize_dates(forecasts, data.date_column)
    actuals = _normalize_dates(actuals, data.date_column)
    if forecasts.duplicated(keys).any():
        raise ValueError("forecasts dataset contains duplicate sku/date rows")
    merged = actuals.merge(forecasts[required], on=keys, how="left", validate="one_to_one")
    uncovered = int(merged["forecast"].isna().sum())
    if uncovered:
        raise ValueError(
            f"forecasts dataset is missing {uncovered} of {len(merged)} requested sku/date rows"
        )
    values = merged["forecast"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("forecast values must all be finite")
    if (values < 0).any():
        raise ValueError("forecast values must be non-negative")
    metrics = forecasting_metrics(merged[data.target_column].to_numpy(dtype=float), values)
    spec = load_task_spec(config)
    if spec is not None:
        frame = merged.rename(columns={data.target_column: "actual"})
        try:
            metrics[spec.name] = compile_metric(spec.code)(frame)
        except Exception as exc:
            raise ValueError(f"custom metric '{spec.name}' failed: {exc}") from exc
    return metrics, merged


def _train_on_foundry(
    worktree: Path,
    config: TaskConfig,
    branch: str,
    *,
    push: bool,
    on_progress: OnProgress | None,
) -> tuple[pd.DataFrame, float]:
    """Push (optionally), build on ``branch``, and return the forecasts frame."""
    fdry = config.foundry
    assert fdry is not None
    expect_new_code = False
    if push:
        if on_progress is not None:
            on_progress(f"pushing code to Foundry branch {branch}")
        expect_new_code = _push_head(worktree, branch, force=branch != fdry.branch)
    started = time.monotonic()
    build_rid = _start_build(config, branch, expect_new_code=expect_new_code, on_progress=on_progress)
    if build_rid is not None:
        if on_progress is not None:
            on_progress("Foundry build queued")
        wait_for_build(
            build_rid,
            timeout_s=fdry.build_timeout_s,
            poll_s=fdry.poll_s,
            on_status=(
                (lambda status, elapsed: on_progress(f"Foundry build {status} ({int(elapsed)}s)"))
                if on_progress is not None
                else None
            ),
        )
    build_s = time.monotonic() - started
    if on_progress is not None:
        on_progress("reading forecasts back from Foundry")
    forecasts = foundry.read_dataset(fdry.datasets.forecasts, branch=branch)
    return forecasts, build_s


def _evaluate_sync(
    worktree: Path,
    config: TaskConfig,
    baseline: dict[str, float] | None,
    guardrails: list[str] | None,
    on_progress: OnProgress | None,
    *,
    include_holdout: bool,
    push: bool,
    branch: str | None,
) -> Evaluation:
    fdry = config.foundry
    assert fdry is not None
    started = time.monotonic()
    branch = branch or _current_branch(worktree) or fdry.branch
    try:
        forecasts, build_s = _train_on_foundry(
            worktree, config, branch, push=push, on_progress=on_progress
        )
        actuals = foundry.read_dataset(fdry.datasets.validation_actuals, branch=fdry.branch)
        metrics, validation_frame = _score_split(forecasts, actuals, config)
        metrics["runtime_s"] = build_s
        expressions = guardrails if guardrails is not None else config.guardrails
        checks = check_guardrails(expressions, metrics, baseline or metrics)
        failures = [f"{item.expression}: {item.detail}" for item in checks if not item.passed]
        if not include_holdout:
            return Evaluation(
                passed=not failures,
                metrics=metrics,
                guardrail_failures=failures,
                duration_s=time.monotonic() - started,
                validation_frame=validation_frame,
            )
        holdout_actuals = foundry.read_dataset(fdry.datasets.holdout_actuals, branch=fdry.branch)
        holdout, _ = _score_split(forecasts, holdout_actuals, config)
        holdout["runtime_s"] = build_s
        primary = config.metric.name
        incumbent_holdout = (baseline or {}).get(f"holdout_{primary}")
        holdout_worse = incumbent_holdout is not None and (
            holdout[primary] > incumbent_holdout
            if config.metric.direction == "min"
            else holdout[primary] < incumbent_holdout
        )
        if holdout_worse:
            failures.append(
                f"hidden holdout {primary} {holdout[primary]:.6f} is worse than incumbent "
                f"{incumbent_holdout:.6f}"
            )
        return Evaluation(
            passed=not failures,
            metrics=metrics,
            holdout_metrics=holdout,
            guardrail_failures=failures,
            duration_s=time.monotonic() - started,
            validation_frame=validation_frame,
        )
    except Exception as exc:  # noqa: BLE001 - evaluator failures become scored failures
        return Evaluation(False, error=str(exc), duration_s=time.monotonic() - started)


async def evaluate_validation(
    worktree: Path,
    config: TaskConfig,
    baseline: dict[str, float] | None = None,
    guardrails: list[str] | None = None,
    on_progress: OnProgress | None = None,
) -> Evaluation:
    """Score the validation split via a Foundry build on the worktree's branch."""
    return await asyncio.to_thread(
        _evaluate_sync,
        worktree,
        config,
        baseline,
        guardrails,
        on_progress,
        include_holdout=False,
        push=True,
        branch=None,
    )


async def evaluate(
    worktree: Path,
    config: TaskConfig,
    baseline: dict[str, float] | None = None,
    guardrails: list[str] | None = None,
    on_progress: OnProgress | None = None,
    *,
    push: bool = True,
    branch: str | None = None,
) -> Evaluation:
    """Score validation + holdout. ``push=False`` scores whatever is currently
    published on ``branch`` (used for the baseline: production as-is)."""
    return await asyncio.to_thread(
        _evaluate_sync,
        worktree,
        config,
        baseline,
        guardrails,
        on_progress,
        include_holdout=True,
        push=push,
        branch=branch,
    )
