from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenCodeResult:
    returncode: int
    session_id: str | None
    error: str | None


async def _shutdown(process: asyncio.subprocess.Process) -> None:
    """Terminate the agent, escalating to SIGKILL; never leave it running."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5)
    except (TimeoutError, asyncio.CancelledError):
        process.kill()
        with contextlib.suppress(asyncio.CancelledError):
            await process.wait()


async def run_opencode(
    cwd: Path,
    prompt: str,
    model: str,
    timeout_s: int,
    log_path: Path,
    on_event: Callable[[dict], None] | None = None,
    *,
    session_id: str | None = None,
    data_dir: Path | None = None,
) -> OpenCodeResult:
    """Run one opencode agent session, streaming its JSON events.

    Every parsed event is appended to ``log_path`` and passed to ``on_event``
    (used by the live dashboard). Cancelling the surrounding task terminates
    the opencode subprocess (then kills it if needed), keeps the JSONL log
    written so far, and re-raises the cancellation.

    Pass ``session_id`` to resume an earlier session with a follow-up prompt
    (the worker's build-evaluate-fix loop). Resuming requires the same
    ``data_dir``: session state lives in the XDG data root, so callers that
    plan to resume must provide a directory that outlives this call (and are
    responsible for deleting it - it holds a copy of auth.json).
    """
    executable = shutil.which("opencode")
    if executable is None:
        return OpenCodeResult(127, None, "opencode is not installed or not on PATH")
    # --dir pins opencode's project directory to cwd. Without it opencode
    # resolves the project by walking up the tree; an agent worktree has only a
    # .git *file* and (for Foundry tasks) sits inside the user's real repo, so
    # opencode anchored there - file writes then landed in the real repo, the
    # worktree snapshot saw no changes, and the worktree opencode.json
    # permissions were never even read.
    command = [
        executable,
        "run",
        "--format",
        "json",
        "--auto",
        "--model",
        model,
        "--dir",
        str(cwd),
    ]
    if session_id:
        command += ["--session", session_id]
    else:
        command += ["--title", f"autoresearch-{cwd.name}"]
    command.append(prompt)
    base_env = os.environ.copy()
    shared_data = Path(
        base_env.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    )
    error: str | None = None
    diagnostics: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCode stores every session in one global SQLite database with no busy
    # timeout. Parallel `opencode run` processes therefore fail immediately
    # with "database is locked". Give each worker isolated XDG data/state roots
    # while keeping config and caches shared. Copy auth.json when present for
    # users who authenticated via the OpenCode CLI rather than provider
    # environment variables.
    async def _launch(isolated: Path) -> OpenCodeResult:
        nonlocal error, diagnostics
        seen_session_id = session_id
        data_home = isolated / "data"
        state_home = isolated / "state"
        auth_source = shared_data / "opencode" / "auth.json"
        if auth_source.is_file():
            auth_target = data_home / "opencode" / "auth.json"
            auth_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(auth_source, auth_target)
        state_home.mkdir(parents=True, exist_ok=True)
        env = base_env | {
            "XDG_DATA_HOME": str(data_home),
            "XDG_STATE_HOME": str(state_home),
        }
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # opencode emits one JSON event per line and single events (tool results,
            # snapshots) can far exceed asyncio's 64 KiB default line limit.
            limit=16 * 1024 * 1024,
        )
        try:
            # Append so every session of one attempt lands in the same log.
            with log_path.open("a") as log:
                assert process.stdout is not None
                async with asyncio.timeout(timeout_s):
                    async for raw in process.stdout:
                        text = raw.decode(errors="replace")
                        log.write(text)
                        log.flush()
                        try:
                            event = json.loads(text)
                        except json.JSONDecodeError:
                            plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).strip()
                            if plain:
                                diagnostics.append(plain)
                                diagnostics = diagnostics[-5:]
                            continue
                        seen_session_id = (
                            event.get("sessionID")
                            or event.get("session_id")
                            or seen_session_id
                        )
                        if on_event is not None:
                            # A display bug must never take down the agent itself.
                            with contextlib.suppress(Exception):
                                on_event(event)
                    await process.wait()
        except TimeoutError:
            await _shutdown(process)
            error = f"opencode exceeded {timeout_s}s timeout"
        except asyncio.CancelledError:
            # Cooperative cancellation: stop the agent, keep the log written so far.
            await _shutdown(process)
            raise
        finally:
            # Never leave an opencode agent running if the read loop failed.
            if process.returncode is None:
                process.kill()
                with contextlib.suppress(asyncio.CancelledError):
                    await process.wait()
        if process.returncode and error is None:
            detail = " · ".join(diagnostics)
            error = f"opencode exited with status {process.returncode}"
            if detail:
                error += f": {detail[:500]}"
        return OpenCodeResult(process.returncode or 0, seen_session_id, error)

    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        return await _launch(data_dir)
    with tempfile.TemporaryDirectory(prefix="autoresearch-opencode-") as scratch:
        return await _launch(Path(scratch))
