from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OpenCodeResult:
    returncode: int
    session_id: str | None
    error: str | None
    attempts: int = 1


def build_opencode_command(
    executable: str, cwd: Path, prompt: str, model: str
) -> list[str]:
    return [
        executable,
        "run",
        "--format",
        "json",
        "--auto",
        "--model",
        model,
        "--dir",
        str(cwd),
        "--title",
        f"autoresearch-{cwd.name}",
        prompt,
    ]


async def run_opencode(
    cwd: Path,
    prompt: str,
    model: str,
    timeout_s: int,
    log_path: Path,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> OpenCodeResult:
    executable = shutil.which("opencode")
    if executable is None:
        return OpenCodeResult(127, None, "opencode is not installed or not on PATH")
    command = build_opencode_command(executable, cwd, prompt, model)
    env = os.environ.copy()
    session_id: str | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = 3
    for launch_attempt in range(1, max_attempts + 1):
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        error: str | None = None
        last_plain_line = ""
        try:
            mode = "w" if launch_attempt == 1 else "a"
            with log_path.open(mode) as log:
                if launch_attempt > 1:
                    log.write(f"\n# OpenCode retry {launch_attempt}/{max_attempts}\n")
                assert process.stdout is not None
                async with asyncio.timeout(timeout_s):
                    async for raw in process.stdout:
                        text = raw.decode(errors="replace")
                        log.write(text)
                        log.flush()
                        try:
                            event = json.loads(text)
                            session_id = (
                                event.get("sessionID")
                                or event.get("session_id")
                                or session_id
                            )
                            if event.get("type") == "error":
                                details = event.get("error") or {}
                                data = details.get("data") if isinstance(details, dict) else {}
                                error = str(
                                    (data or {}).get("message")
                                    or (details or {}).get("name")
                                    or "OpenCode error"
                                )
                            if on_event:
                                with suppress(Exception):
                                    on_event(event)
                        except json.JSONDecodeError:
                            if text.strip():
                                last_plain_line = text.strip()
                    await process.wait()
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
            return OpenCodeResult(
                process.returncode or 1,
                session_id,
                f"opencode exceeded {timeout_s}s timeout",
                launch_attempt,
            )

        returncode = process.returncode or 0
        if returncode == 0:
            return OpenCodeResult(0, session_id, None, launch_attempt)
        error = error or last_plain_line or f"opencode exited with status {returncode}"
        if launch_attempt < max_attempts:
            if on_event:
                with suppress(Exception):
                    on_event(
                        {
                            "type": "runtime_retry",
                            "attempt": launch_attempt + 1,
                            "max_attempts": max_attempts,
                        }
                    )
            await asyncio.sleep(2 ** (launch_attempt - 1))
    return OpenCodeResult(returncode, session_id, error, max_attempts)
