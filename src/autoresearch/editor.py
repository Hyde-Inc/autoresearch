"""Best-effort opening of notebook files in the user's editor.

The CLI runs in a terminal, but plan and findings files are meant to be read
and edited in an editor. Whenever the flow produces a file the human should be
looking at, we pop it open: Cursor or VS Code if their CLI is on PATH, the
Cursor app bundle on macOS, then the OS default opener. Set
AUTORESEARCH_NO_OPEN=1 to disable (scripted tests do this).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_MAC_CURSOR_CLI = Path("/Applications/Cursor.app/Contents/Resources/app/bin/cursor")


def open_in_editor(path: Path) -> bool:
    """Open *path* in the user's editor without blocking. Returns True if launched."""
    if os.environ.get("AUTORESEARCH_NO_OPEN"):
        return False
    candidates: list[list[str]] = []
    for name in ("cursor", "code"):
        exe = shutil.which(name)
        if exe:
            candidates.append([exe, "--reuse-window", str(path)])
    if sys.platform == "darwin":
        if _MAC_CURSOR_CLI.exists():
            candidates.append([str(_MAC_CURSOR_CLI), "--reuse-window", str(path)])
        candidates.append(["open", str(path)])
    for command in candidates:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            continue
        return True
    return False
