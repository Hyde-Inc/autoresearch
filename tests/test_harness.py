import asyncio
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from autoresearch.config import load_config
from autoresearch.harness import (
    _evaluate_split,
    evaluate,
    evaluate_validation,
    forecasting_metrics,
)


def test_forecasting_metrics() -> None:
    metrics = forecasting_metrics(
        np.array([10.0, 20.0, 30.0]), np.array([11.0, 18.0, 33.0])
    )
    assert metrics["wmape"] == 0.1
    assert metrics["rmse"] > 0
    assert metrics["bias_pct"] > 0


def _fake_split(splits_seen: list[str], wmape: float = 0.09):
    async def fake(worktree, actuals_path, config, label, on_progress=None):
        splits_seen.append(label)
        frame = pd.DataFrame({"actual": [1.0], "forecast": [1.0]})
        return {"wmape": wmape, "mape": 0.1, "rmse": 1.0, "bias_pct": 0.0}, 1.5, frame

    return fake


def test_evaluate_validation_never_touches_the_holdout_split(tmp_path: Path, monkeypatch) -> None:
    """The worker loop iterates against this feedback signal, so it must not
    leak the hidden holdout split."""
    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "task.yaml").write_text(yaml.safe_dump({"name": "t", "goal": "g"}))
    config = load_config(task_root / "task.yaml")
    splits: list[str] = []
    monkeypatch.setattr("autoresearch.harness._evaluate_split", _fake_split(splits))
    result = asyncio.run(
        evaluate_validation(tmp_path, config, baseline={"wmape": 0.10}, guardrails=[])
    )
    assert splits == ["validation"]
    assert result.passed
    assert result.metrics["wmape"] == 0.09
    assert result.metrics["runtime_s"] == 1.5
    assert result.holdout_metrics == {}


def test_evaluate_runs_both_splits_and_keeps_the_holdout_gate(
    tmp_path: Path, monkeypatch
) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "task.yaml").write_text(yaml.safe_dump({"name": "t", "goal": "g"}))
    config = load_config(task_root / "task.yaml")
    splits: list[str] = []
    monkeypatch.setattr("autoresearch.harness._evaluate_split", _fake_split(splits))
    result = asyncio.run(
        evaluate(
            tmp_path,
            config,
            # The candidate's holdout (0.09) is worse than the incumbent's 0.05.
            baseline={"wmape": 0.10, "holdout_wmape": 0.05},
            guardrails=[],
        )
    )
    assert splits == ["validation", "holdout"]
    assert not result.passed
    assert any("hidden holdout" in failure for failure in result.guardrail_failures)
    assert result.holdout_metrics["wmape"] == 0.09


def test_training_output_streams_live_and_failures_keep_the_tail(
    tmp_path: Path, monkeypatch
) -> None:
    """train.py output (including carriage-return progress bars) must reach the
    on_progress callback while it runs, and the failure message must carry the
    output tail."""
    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "task.yaml").write_text(yaml.safe_dump({"name": "t", "goal": "g"}))
    config = load_config(task_root / "task.yaml")
    actuals = tmp_path / "validation.parquet"
    pd.DataFrame(
        {"sku": ["a"], "date": pd.to_datetime(["2025-01-01"]), "demand": [1.0]}
    ).to_parquet(actuals, index=False)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    real_exec = asyncio.create_subprocess_exec

    async def fake_train(*args, **kwargs):
        return await real_exec(
            "/bin/sh",
            "-c",
            "printf 'epoch 1/3\\repoch 2/3\\repoch 3/3\\nsaving model\\n'; "
            "echo 'boom traceback' >&2; exit 3",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_train)
    lines: list[str] = []
    with pytest.raises(ValueError, match="solution exited with 3") as excinfo:
        asyncio.run(
            _evaluate_split(worktree, actuals, config, "validation", on_progress=lines.append)
        )
    assert lines[:4] == ["epoch 1/3", "epoch 2/3", "epoch 3/3", "saving model"]
    assert "boom traceback" in lines
    assert "boom traceback" in str(excinfo.value)
    assert not (worktree / ".autoresearch-validation-request.parquet").exists()


def test_cancellation_kills_training_and_stays_a_cancellation(tmp_path: Path, monkeypatch) -> None:
    """Interrupting the protected evaluator must kill train.py, remove the
    request file, and re-raise the cancellation instead of scoring a failure."""
    task_root = tmp_path / "task"
    task_root.mkdir()
    (task_root / "task.yaml").write_text(yaml.safe_dump({"name": "t", "goal": "g"}))
    config = load_config(task_root / "task.yaml")
    actuals = tmp_path / "validation.parquet"
    pd.DataFrame(
        {"sku": ["a", "b"], "date": pd.to_datetime(["2025-01-01"] * 2), "demand": [1.0, 2.0]}
    ).to_parquet(actuals, index=False)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    pid_file = tmp_path / "train.pid"

    real_exec = asyncio.create_subprocess_exec

    async def fake_train(*args, **kwargs):
        # Stand-in for `uv run ... solution/train.py`: a long-running process.
        return await real_exec(
            "/bin/sh",
            "-c",
            f"echo $$ > {pid_file}; exec sleep 30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_train)

    async def scenario() -> None:
        task = asyncio.create_task(_evaluate_split(worktree, actuals, config, "validation"))
        deadline = time.monotonic() + 5.0
        while not pid_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5.0
    dead = False
    while time.monotonic() < deadline and not dead:
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            dead = True
    assert dead, "train.py subprocess survived cancellation"
    assert not (worktree / ".autoresearch-validation-request.parquet").exists()
