"""Round execution under the dashboard: per-agent cancel, round cancel, cleanup.

run_worker and evaluate are mocked; cancellation flows through the real
RoundController / _execute_attempt / _run_round machinery.
"""

import asyncio
import threading
import time
from pathlib import Path

import yaml

from autoresearch.config import load_config
from autoresearch.harness import Evaluation
from autoresearch.models import Idea
from autoresearch.opencode import OpenCodeResult
from autoresearch.orchestrator import _run_round
from autoresearch.store import RunStore
from autoresearch.worker import WorkerResult


def _config(tmp_path: Path):
    task_root = tmp_path / "repo" / ".autoresearch" / "task"
    task_root.mkdir(parents=True)
    (task_root / "task.yaml").write_text(yaml.safe_dump({"name": "t", "goal": "reduce wmape"}))
    return load_config(task_root / "task.yaml")


def _ideas(*titles: str) -> list[Idea]:
    return [Idea(title=title, hypothesis="h", instructions="i") for title in titles]


def _install_fakes(monkeypatch, store: RunStore, *, slow_titles: set[str], sleep_s: float = 30.0):
    """run_worker creates a real worktree dir then sleeps if the idea is slow;
    evaluate always passes."""

    async def fake_run_worker(*, config, store, repo, idea, attempt_id, base_ref, baseline=None, on_event=None, on_phase=None):
        worktree = store.worktrees_dir / attempt_id
        worktree.mkdir(parents=True, exist_ok=True)
        if on_phase is not None:
            on_phase("Exploring", "coding agent starting")
        if idea.title in slow_titles:
            await asyncio.sleep(sleep_s)
        else:
            await asyncio.sleep(0.4)
        return WorkerResult(
            branch=f"autoresearch/{attempt_id}",
            worktree=worktree,
            commit="c0ffee",
            diff="diff",
            changed_paths=["solution/train.py"],
            opencode=OpenCodeResult(0, "ses_x", None),
        )

    async def fake_evaluate(worktree, config, baseline=None, guardrails=None, on_progress=None):
        return Evaluation(
            passed=True,
            metrics={"wmape": 0.09},
            holdout_metrics={"wmape": 0.10},
            duration_s=0.01,
        )

    monkeypatch.setattr("autoresearch.orchestrator.run_worker", fake_run_worker)
    monkeypatch.setattr("autoresearch.orchestrator.evaluate", fake_evaluate)


class _CancelSecondAgentDashboard:
    """Stub dashboard that presses `x` on row 2 shortly after the round starts,
    exercising the real thread-safe cancel path."""

    def __init__(self, console, header, agents, *, on_cancel_agent=None, on_cancel_round=None, **_):
        self.on_cancel_agent = on_cancel_agent

    def __enter__(self):
        threading.Timer(0.15, self.on_cancel_agent, args=(1,)).start()
        return self

    def __exit__(self, *args):
        pass


def test_cancel_one_agent_while_siblings_complete(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "run")
    repo = tmp_path / "repo"
    _install_fakes(monkeypatch, store, slow_titles={"slow"})
    monkeypatch.setattr("autoresearch.orchestrator.AgentDashboard", _CancelSecondAgentDashboard)

    results, round_cancelled = asyncio.run(
        _run_round(
            config,
            store,
            repo,
            _ideas("fast one", "slow", "fast two"),
            1,
            "main",
            {"wmape": 0.1039},
            "Round 1",
        )
    )
    assert not round_cancelled
    statuses = [attempt.status for attempt, _ in results]
    assert statuses == ["passed", "cancelled", "passed"]
    # Siblings kept running after the cancel and their results survive.
    assert all(attempt.metrics["wmape"] == 0.09 for attempt, _ in (results[0], results[2]))
    cancelled_attempt, cancelled_worker = results[1]
    assert cancelled_worker is None
    assert cancelled_attempt.error == "cancelled by user"
    # Persistence: the run store recorded the interrupted attempt as cancelled.
    stored = {item.id: item.status for item in store.load_attempts()}
    assert stored[cancelled_attempt.id] == "cancelled"
    # The cancelled agent's worktree was cleaned; the finished ones remain
    # until the orchestrator removes them after promotion.
    assert not (store.worktrees_dir / cancelled_attempt.id).exists()
    for attempt, worker in (results[0], results[2]):
        assert worker is not None and worker.worktree.exists()


def test_stop_flag_cancels_the_whole_round_immediately(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    store = RunStore(tmp_path / "run")
    repo = tmp_path / "repo"
    _install_fakes(monkeypatch, store, slow_titles={"a", "b", "c"})
    monkeypatch.chdir(tmp_path)
    Path(".autoresearch-stop").write_text("stop\n")

    started = time.monotonic()
    results, round_cancelled = asyncio.run(
        _run_round(
            config,
            store,
            repo,
            _ideas("a", "b", "c"),
            1,
            "main",
            {"wmape": 0.1039},
            "Round 1",
            stop_poll_s=0.05,
        )
    )
    elapsed = time.monotonic() - started
    assert round_cancelled
    assert elapsed < 10, "stop flag did not interrupt the round promptly"
    assert [attempt.status for attempt, _ in results] == ["cancelled"] * 3
    # Every agent's worktree was cleaned on the way out.
    assert not any(store.worktrees_dir.iterdir())
    assert all(item.status == "cancelled" for item in store.load_attempts())
