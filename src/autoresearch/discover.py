"""Repo discovery: inventory a data science project so the director can explore it.

The director never asks the user where files live; it reads this inventory and
uses read_file / inspect_data tools to find training data, existing models, and
project context on its own.
"""

from __future__ import annotations

from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".autoresearch",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    ".idea",
    ".vscode",
    "dist",
    "research",  # the human-facing lab notebook; keep it out of agent workspaces
}
DATA_SUFFIXES = {".csv", ".parquet", ".pq"}
CODE_SUFFIXES = {".py", ".ipynb", ".sql", ".r"}
DOC_NAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}
MAX_FILES = 400


def _walk(repo: Path) -> list[Path]:
    found: list[Path] = []
    stack = [repo]
    while stack and len(found) < MAX_FILES:
        current = stack.pop()
        for entry in sorted(current.iterdir()):
            if entry.name.startswith(".") and entry.name not in DOC_NAMES:
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS:
                    stack.append(entry)
            else:
                found.append(entry)
                if len(found) >= MAX_FILES:
                    break
    return found


def repo_inventory(repo: Path) -> dict:
    repo = repo.resolve()
    data_files, code_files, docs, other = [], [], [], []
    for path in _walk(repo):
        relative = str(path.relative_to(repo))
        suffix = path.suffix.lower()
        if suffix in DATA_SUFFIXES:
            data_files.append({"path": relative, "bytes": path.stat().st_size})
        elif suffix in CODE_SUFFIXES:
            code_files.append(relative)
        elif path.name.lower() in DOC_NAMES or suffix in {".md", ".toml", ".yaml", ".yml", ".txt"}:
            docs.append(relative)
        else:
            other.append(relative)
    return {
        "root": str(repo),
        "data_files": data_files,
        "code_files": code_files,
        "docs_and_config": docs,
        "other_files": other[:40],
    }


def render_inventory(inventory: dict) -> str:
    lines = [f"Project root: {inventory['root']}"]
    sections = [
        ("Data files", [f"{i['path']} ({i['bytes']:,} bytes)" for i in inventory["data_files"]]),
        ("Code files", inventory["code_files"]),
        ("Docs and config", inventory["docs_and_config"]),
    ]
    for title, entries in sections:
        lines.append(f"{title}:")
        if entries:
            lines.extend(f"  {entry}" for entry in entries)
        else:
            lines.append("  (none found)")
    return "\n".join(lines)


def read_repo_file(repo: Path, relative: str, max_chars: int = 12000) -> str:
    repo = repo.resolve()
    target = (repo / relative).resolve()
    if repo not in target.parents and target != repo:
        raise ValueError(f"path escapes the project: {relative}")
    if not target.is_file():
        raise FileNotFoundError(f"no such file in the project: {relative}")
    text = target.read_text(errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... truncated ({len(text)} chars total)"
    return text
