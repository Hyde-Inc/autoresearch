from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenCodeResult:
    returncode: int
    session_id: str | None
    error: str | None


async def run_opencode(
    cwd: Path,
    prompt: str,
    model: str,
    timeout_s: int,
    log_path: Path,
) -> OpenCodeResult:
    executable = shutil.which("opencode")
    if executable is None:
        return OpenCodeResult(127, None, "opencode is not installed or not on PATH")
    command = [
        executable,
        "run",
        "--format",
        "json",
        "--auto",
        "--model",
        model,
        "--title",
        f"autoresearch-{cwd.name}",
        prompt,
    ]
    env = os.environ.copy()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    session_id: str | None = None
    error: str | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as log:
            assert process.stdout is not None
            async with asyncio.timeout(timeout_s):
                async for raw in process.stdout:
                    text = raw.decode(errors="replace")
                    log.write(text)
                    log.flush()
                    try:
                        event = json.loads(text)
                        session_id = event.get("sessionID") or event.get("session_id") or session_id
                    except json.JSONDecodeError:
                        pass
                await process.wait()
    except TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()
        error = f"opencode exceeded {timeout_s}s timeout"
    return OpenCodeResult(process.returncode or 0, session_id, error)
