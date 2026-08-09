"""Repo understanding via opencode: a coding agent surveys the project once.

Instead of the director piecing the repo together file by file, an opencode
agent explores a throwaway copy of the project and writes a structured
SURVEY.md: baseline models and entry points, training/eval data and columns,
evaluation conventions, and dependencies. The findings are cached under
``<repo>/.autoresearch/survey.md`` and fed into the director's context. When
opencode is unavailable or fails, callers fall back to the static inventory.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from .discover import SKIP_DIRS
from .opencode import run_opencode

SURVEY_CACHE = ".autoresearch/survey.md"
SURVEY_TIMEOUT_S = 300
MAX_SURVEY_CHARS = 12000

_PROMPT = """\
You are surveying a data science repository so an ML research director can plan
experiments against it. Investigate the project - read the README, the code,
and inspect data files (you may run small Python snippets to peek at columns
and date ranges) - then write your findings to a file named SURVEY.md in the
repository root. Use exactly these sections:

# Repository survey
## Overview
What the project does, in a few sentences.
## Models
Every model or training script: path, what it implements, how it is run, and
which one looks like the production/incumbent baseline.
## Data
Every data file relevant to training or evaluation: path, format, columns with
their meaning (id / date / target / covariates), row counts, and date ranges.
## Evaluation
How the project evaluates forecasts today: metrics, backtest windows, horizons.
## Dependencies
The libraries the project relies on.

Be factual and specific: exact paths, exact column names. Do not modify any
other file. Write SURVEY.md and stop."""


def cached_survey(repo: Path) -> str | None:
    cache = repo / SURVEY_CACHE
    if cache.is_file():
        return cache.read_text()[:MAX_SURVEY_CHARS]
    return None


def _copy_repo(repo: Path, destination: Path) -> None:
    shutil.copytree(
        repo,
        destination,
        ignore=shutil.ignore_patterns(*SKIP_DIRS, ".*"),
        dirs_exist_ok=True,
    )


async def _survey(repo: Path, model: str, timeout_s: int, log_path: Path) -> str | None:
    with tempfile.TemporaryDirectory(prefix="autoresearch-survey-") as scratch:
        workspace = Path(scratch) / repo.name
        _copy_repo(repo, workspace)
        result = await run_opencode(workspace, _PROMPT, model, timeout_s, log_path)
        report = workspace / "SURVEY.md"
        if not report.is_file():
            return None
        text = report.read_text().strip()
        if result.error and not text:
            return None
        return text[:MAX_SURVEY_CHARS] or None


def survey_repo(
    repo: Path,
    model: str,
    timeout_s: int = SURVEY_TIMEOUT_S,
    refresh: bool = False,
) -> str | None:
    """Return the survey report for a repo, running opencode on a copy if needed.

    Returns None when opencode is unavailable or produced nothing; callers fall
    back to the static repo inventory.
    """
    repo = repo.resolve()
    if not refresh:
        cached = cached_survey(repo)
        if cached:
            return cached
    log_path = repo / ".autoresearch" / "logs" / "survey.jsonl"
    text = asyncio.run(_survey(repo, model, timeout_s, log_path))
    if text:
        cache = repo / SURVEY_CACHE
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    return text
