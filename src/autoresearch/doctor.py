"""Preflight checks: confirm a machine can actually run a research loop.

``run_checks`` verifies the external tools the pipeline shells out to (git for
worktrees, uv to run experiments, opencode as the coding agent) and the
credentials it reads from the environment (OpenRouter for the models, Foundry
for optional ingestion). It is intentionally dependency-light - a live
OpenRouter probe uses urllib so ``autoresearch doctor`` works before anything
else is configured.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

Status = Literal["ok", "warn", "fail"]

MIN_PYTHON = (3, 12)


@dataclass
class Check:
    name: str
    status: Status
    detail: str


def _tool_version(executable: str) -> str:
    """First line of ``<tool> --version``, or an empty string if it won't run."""
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else ""


def _check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] < MIN_PYTHON:
        return Check(
            "python",
            "fail",
            f"{version} found; the experiments require Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        )
    return Check("python", "ok", version)


def _check_tool(executable: str, purpose: str) -> Check:
    path = shutil.which(executable)
    if path is None:
        return Check(executable, "fail", f"not on PATH - required to {purpose}")
    version = _tool_version(executable)
    detail = f"{version} ({path})" if version else path
    return Check(executable, "ok", detail)


def _check_dotenv() -> Check:
    try:
        from dotenv import find_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency
        return Check(".env", "warn", "python-dotenv not installed")
    found = find_dotenv(usecwd=True)
    if found:
        return Check(".env", "ok", found)
    return Check(
        ".env",
        "warn",
        "no .env found from the current directory; relying on exported environment variables",
    )


def _probe_openrouter(api_key: str) -> Check:
    """Ask OpenRouter to validate the key and report remaining credit."""
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return Check("openrouter", "fail", f"key rejected by OpenRouter (HTTP {exc.code})")
        return Check("openrouter", "warn", f"could not verify key (HTTP {exc.code})")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return Check("openrouter", "warn", f"key is set but OpenRouter was unreachable: {reason}")

    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    usage = data.get("usage")
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    if remaining is not None:
        detail = f"key valid; {remaining} credit remaining"
    elif limit is None:
        detail = f"key valid; unlimited credit (usage ${usage or 0:.4f})"
    else:
        detail = f"key valid; used ${usage or 0:.4f} of ${limit}"
    return Check("openrouter", "ok", detail)


def _check_openrouter(check_api: bool) -> Check:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return Check("openrouter", "fail", "OPENROUTER_API_KEY is not set (add it to .env)")
    if not check_api:
        return Check("openrouter", "ok", "OPENROUTER_API_KEY is set (skipped live check)")
    return _probe_openrouter(api_key)


def _check_foundry() -> Check:
    hostname = os.getenv("FOUNDRY_HOSTNAME", "").strip()
    token = os.getenv("FOUNDRY_TOKEN", "").strip()
    if hostname and token:
        return Check("foundry", "ok", f"credentials set for {hostname}")
    if hostname or token:
        missing = "FOUNDRY_TOKEN" if hostname else "FOUNDRY_HOSTNAME"
        return Check("foundry", "warn", f"{missing} is missing; Foundry ingestion will fail")
    return Check("foundry", "warn", "not configured (only needed for Foundry ingestion)")


def run_checks(*, check_api: bool = True) -> list[Check]:
    """Run every preflight check. ``check_api`` toggles the live OpenRouter probe."""
    return [
        _check_python(),
        _check_tool("git", "create experiment worktrees"),
        _check_tool("uv", "run experiment training"),
        _check_tool("opencode", "run the coding agent"),
        _check_dotenv(),
        _check_openrouter(check_api),
        _check_foundry(),
    ]
