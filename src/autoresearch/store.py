from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .models import Attempt


class RunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.attempts_dir = self.run_dir / "attempts"
        self.logs_dir = self.run_dir / "logs"
        self.worktrees_dir = self.run_dir / "worktrees"
        for path in (self.run_dir, self.attempts_dir, self.logs_dir, self.worktrees_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.attempts_file = self.run_dir / "attempts.jsonl"
        self.notes_file = self.run_dir / "notes.md"
        self.state_file = self.run_dir / "state.json"

    @classmethod
    def create(cls, base: Path, task_name: str) -> RunStore:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return cls(base.resolve() / task_name / stamp)

    def save_attempt(self, attempt: Attempt) -> None:
        target = self.attempts_dir / f"{attempt.id}.json"
        payload = attempt.model_dump(mode="json")
        target.write_text(json.dumps(payload, indent=2) + "\n")
        attempts = {item.id: item for item in self.load_attempts()}
        attempts[attempt.id] = attempt
        ordered = sorted(attempts.values(), key=lambda item: item.created_at)
        self.attempts_file.write_text(
            "".join(json.dumps(item.model_dump(mode="json")) + "\n" for item in ordered)
        )

    def load_attempts(self) -> list[Attempt]:
        if not self.attempts_file.exists():
            return []
        result = []
        for line in self.attempts_file.read_text().splitlines():
            if line.strip():
                result.append(Attempt.model_validate_json(line))
        return result

    def leaderboard(self, metric: str, direction: str = "min") -> list[Attempt]:
        scored = [
            item
            for item in self.load_attempts()
            if item.status == "passed" and metric in item.metrics
        ]
        return sorted(scored, key=lambda item: item.metrics[metric], reverse=direction == "max")

    def save_validation_frame(self, attempt_id: str, frame: pd.DataFrame) -> Path:
        target = self.attempts_dir / f"{attempt_id}-validation.parquet"
        frame.to_parquet(target, index=False)
        return target

    def load_validation_frame(self, attempt_id: str) -> pd.DataFrame | None:
        target = self.attempts_dir / f"{attempt_id}-validation.parquet"
        return pd.read_parquet(target) if target.exists() else None

    def list_validation_frames(self) -> list[str]:
        return sorted(
            path.name.removesuffix("-validation.parquet")
            for path in self.attempts_dir.glob("*-validation.parquet")
        )

    def append_note(self, heading: str, body: str) -> None:
        with self.notes_file.open("a") as handle:
            handle.write(f"\n## {heading}\n\n{body.strip()}\n")

    def save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2) + "\n")

    def load_state(self) -> dict:
        if not self.state_file.exists():
            return {}
        return json.loads(self.state_file.read_text())


def latest_run(base: Path, task_name: str | None = None) -> Path | None:
    root = base.resolve() / task_name if task_name else base.resolve()
    candidates = [p for p in root.glob("*") if p.is_dir()] if root.exists() else []
    if task_name is None:
        candidates = [run for task in candidates for run in task.glob("*") if run.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
