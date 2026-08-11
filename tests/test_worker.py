"""The worker's build-evaluate-fix loop: feedback, harvesting, and stop rules.

run_opencode and evaluate_validation are mocked; git worktrees, snapshot
commits, and the best-snapshot restore run for real.
"""

import asyncio
from pathlib import Path

import yaml

from autoresearch.config import load_config
from autoresearch.harness import Evaluation
from autoresearch.models import Idea
from autoresearch.opencode import OpenCodeResult
from autoresearch.store import RunStore
from autoresearch.worker import _run, ensure_seed_repo, run_worker


def test_seed_copy_supports_worktrees(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    tree = tmp_path / "worktree"
    (repo / "solution").mkdir(parents=True)
    (repo / "solution" / "train.py").write_text("print('baseline')\n")
    (repo / "pyproject.toml").write_text("[project]\nname='seed'\nversion='0'\n")
    asyncio.run(ensure_seed_repo(repo))
    code, output = asyncio.run(
        _run("git", "worktree", "add", "-b", "autoresearch/test", str(tree), "main", cwd=repo)
    )
    assert code == 0, output
    assert (tree / "solution/train.py").exists()
    asyncio.run(_run("git", "worktree", "remove", "--force", str(tree), cwd=repo))


def _setup(tmp_path: Path, agents: dict | None = None):
    repo = tmp_path / "repo"
    (repo / "solution").mkdir(parents=True)
    (repo / "solution" / "train.py").write_text("print('baseline')\n")
    (repo / "TASK.md").write_text("# task\n")
    asyncio.run(ensure_seed_repo(repo))
    task_root = tmp_path / "task"
    task_root.mkdir()
    raw = {"name": "t", "goal": "reduce wmape"}
    if agents:
        raw["agents"] = agents
    (task_root / "task.yaml").write_text(yaml.safe_dump(raw))
    config = load_config(task_root / "task.yaml")
    store = RunStore(tmp_path / "run")
    return config, store, repo


class _FakeLab:
    """Scripted agent sessions and evaluator verdicts for one run_worker call."""

    def __init__(self, sessions: list[str | None], evaluations: list[Evaluation],
                 results: list[OpenCodeResult] | None = None,
                 progress_lines: list[str] | None = None):
        self.sessions = sessions  # train.py content per session; None = no edits
        self.pending = list(evaluations)  # consumed only when training actually runs
        self.results = results or []
        self.progress_lines = progress_lines or []
        self.eval_calls = 0
        self.prompts: list[str] = []
        self.session_ids: list[str | None] = []
        self.data_dirs: list[Path] = []
        self.experiments: list[str] = []
        self.permissions: list[str | None] = []

    async def run_opencode(self, cwd, prompt, model, timeout_s, log_path,
                           on_event=None, *, session_id=None, data_dir=None):
        index = len(self.prompts)
        self.prompts.append(prompt)
        self.session_ids.append(session_id)
        self.data_dirs.append(data_dir)
        self.experiments.append((cwd / "EXPERIMENT.md").read_text())
        permissions = cwd / "opencode.json"
        self.permissions.append(permissions.read_text() if permissions.exists() else None)
        if data_dir is not None:  # the real runner creates the session store
            data_dir.mkdir(parents=True, exist_ok=True)
        content = self.sessions[index]
        if content is not None:
            (cwd / "solution" / "train.py").write_text(content)
        if index < len(self.results):
            return self.results[index]
        return OpenCodeResult(0, f"ses_{index + 1}" if session_id is None else session_id, None)

    async def evaluate_validation(self, worktree, config, baseline=None, guardrails=None,
                                  on_progress=None):
        self.eval_calls += 1
        if on_progress is not None:
            for line in self.progress_lines:
                on_progress(line)
        return self.pending.pop(0)


def _run_loop(
    monkeypatch,
    lab: _FakeLab,
    config,
    store,
    repo,
    baseline,
    on_phase=None,
    skills_used: list[str] | None = None,
):
    monkeypatch.setattr("autoresearch.worker.run_opencode", lab.run_opencode)
    monkeypatch.setattr("autoresearch.worker.evaluate_validation", lab.evaluate_validation)
    return asyncio.run(
        run_worker(
            config=config,
            store=store,
            repo=repo,
            idea=Idea(
                title="loop test",
                hypothesis="h",
                instructions="i",
                skills_used=skills_used or [],
            ),
            attempt_id="loop0001",
            base_ref="main",
            baseline=baseline,
            on_phase=on_phase,
        )
    )


def test_worker_confines_the_agent_when_the_repo_has_no_opencode_json(
    tmp_path: Path, monkeypatch
) -> None:
    """A Foundry transforms repo carries no opencode.json, so the worker must
    supply one (external_directory=deny keeps the agent out of the run store)
    and clean it up so it never appears in the diff or reaches Foundry."""
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["print('v1')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0})],
    )
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert lab.permissions[0] is not None and '"external_directory": "deny"' in lab.permissions[0]
    assert not (result.worktree / "opencode.json").exists()
    assert "opencode.json" not in result.changed_paths


def test_worker_keeps_a_tracked_opencode_json(tmp_path: Path, monkeypatch) -> None:
    config, store, repo = _setup(tmp_path)
    (repo / "opencode.json").write_text('{"permission": {"external_directory": "deny"}}\n')
    asyncio.run(_run("git", "add", "opencode.json", cwd=repo))
    asyncio.run(_run("git", "commit", "-m", "seed permissions", cwd=repo))
    lab = _FakeLab(
        sessions=["print('v1')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0})],
    )
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    # The seed's own file is used as-is and must survive the cleanup.
    assert (result.worktree / "opencode.json").exists()
    assert "opencode.json" not in result.changed_paths


def test_evaluator_failure_feeds_back_and_the_fix_wins(tmp_path: Path, monkeypatch) -> None:
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["print('v1')\n", "print('v2')\n"],
        evaluations=[
            Evaluation(False, error="solution exited with 1: boom"),
            Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0}),
        ],
    )
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert result.error is None
    assert result.sessions == 2
    assert result.commit is not None
    # The second session resumed the first conversation with the traceback.
    assert lab.session_ids == [None, "ses_1"]
    assert "boom" in lab.prompts[1]
    # The per-session history records which step each session reached.
    assert [(entry["session"], entry["step"]) for entry in result.history] == [
        (1, "train"),
        (2, "scored"),
    ]
    assert result.history[1]["metric"] == 0.08
    # Both sessions share the persistent per-attempt session store...
    assert len({str(path) for path in lab.data_dirs}) == 1
    # ...which is removed afterwards (it holds a copy of auth.json).
    assert not (store.run_dir / "opencode" / "loop0001").exists()
    worktree = store.worktrees_dir / "loop0001"
    assert (worktree / "solution" / "train.py").read_text() == "print('v2')\n"
    assert "solution/train.py" in result.changed_paths
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_timeout_is_harvested_not_fatal(tmp_path: Path, monkeypatch) -> None:
    """The round-1 failure mode: a session that times out mid-flight still has
    its solution/ state committed and scored instead of being thrown away."""
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["print('v1')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0})],
        results=[OpenCodeResult(-15, "ses_1", "opencode exceeded 1200s timeout")],
    )
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert result.error is None
    assert result.commit is not None
    assert result.sessions == 1
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_best_snapshot_is_restored_after_a_regression(tmp_path: Path, monkeypatch) -> None:
    lab = _FakeLab(
        sessions=["print('v1')\n", "print('v2')\n", "print('v3')\n"],
        evaluations=[
            Evaluation(True, metrics={"wmape": 0.09, "runtime_s": 5.0}),
            Evaluation(True, metrics={"wmape": 0.12, "runtime_s": 5.0}),
            Evaluation(True, metrics={"wmape": 0.13, "runtime_s": 5.0}),
        ],
    )
    config, store, repo = _setup(tmp_path)
    # An unbeatable incumbent keeps the loop improving until the stall limit.
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.05})
    assert result.sessions == 3
    assert "Improve the solution" in lab.prompts[1]
    worktree = store.worktrees_dir / "loop0001"
    # Sessions 2 and 3 regressed; the branch ends on the session-1 snapshot.
    assert (worktree / "solution" / "train.py").read_text() == "print('v1')\n"
    _, message = asyncio.run(_run("git", "log", "-1", "--format=%s", cwd=worktree))
    assert "(session 1)" in message
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_identical_failure_twice_stops_the_loop(tmp_path: Path, monkeypatch) -> None:
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["print('v1')\n", "print('v2')\n"],
        evaluations=[
            Evaluation(False, error="forecast values must be non-negative"),
            Evaluation(False, error="forecast values must be non-negative"),
        ],
    )
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert result.sessions == 2
    # The harvested commit survives; the final protected eval reports the error.
    assert result.error is None
    assert result.commit is not None
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_budget_exhaustion_stops_followup_sessions(tmp_path: Path, monkeypatch) -> None:
    config, store, repo = _setup(tmp_path, agents={"budget_s": 1})
    lab = _FakeLab(
        sessions=["print('v1')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.2, "runtime_s": 5.0})],
    )
    # 0.2 does not beat 0.1, but there is no budget left for an improve session.
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert result.sessions == 1
    assert result.commit is not None
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_no_changes_is_still_a_failure(tmp_path: Path, monkeypatch) -> None:
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(sessions=[None], evaluations=[])
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert result.error == "agent made no changes"
    assert result.commit is None
    assert result.sessions == 1
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_build_verification_catches_syntax_errors_before_training(
    tmp_path: Path, monkeypatch
) -> None:
    """A solution that cannot even parse must not cost a training run: the
    build check fails fast and the error goes straight back to the agent."""
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["def broken(:\n", "print('fixed')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0})],
    )
    result = _run_loop(monkeypatch, lab, config, store, repo, {"wmape": 0.10})
    assert lab.eval_calls == 1  # only the fixed session reached training
    assert "syntax error" in lab.prompts[1]
    assert result.error is None
    assert result.sessions == 2
    assert [(entry["session"], entry["step"]) for entry in result.history] == [
        (1, "build"),
        (2, "scored"),
    ]
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_step_phases_and_training_progress_reach_the_dashboard(
    tmp_path: Path, monkeypatch
) -> None:
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["print('v1')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0})],
        progress_lines=["epoch 5/10 loss 0.31"],
    )
    phases: list[tuple[str, str]] = []
    result = _run_loop(
        monkeypatch, lab, config, store, repo, {"wmape": 0.10},
        on_phase=lambda name, action: phases.append((name, action)),
    )
    assert result.error is None
    # Live training output lands on the dashboard as the Training phase action.
    assert ("Training", "epoch 5/10 loss 0.31") in phases
    # The loop walks build -> snapshot -> train -> evaluate in order.
    names = [name for name, _ in phases]
    for step, following in (
        ("Exploring", "Committing"),
        ("Committing", "Training"),
        ("Training", "Protected eval"),
    ):
        assert names.index(following) > names.index(step)
    scored = [action for name, action in phases if name == "Protected eval"]
    assert any("wmape 0.08" in action for action in scored)
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))


def test_worker_receives_full_selected_skill_playbook(tmp_path: Path, monkeypatch) -> None:
    config, store, repo = _setup(tmp_path)
    lab = _FakeLab(
        sessions=["print('v1')\n"],
        evaluations=[Evaluation(True, metrics={"wmape": 0.08, "runtime_s": 5.0})],
    )
    result = _run_loop(
        monkeypatch,
        lab,
        config,
        store,
        repo,
        {"wmape": 0.10},
        skills_used=["boosting-demand-models"],
    )
    assert result.error is None
    experiment = lab.experiments[0]
    assert "## Implementation playbooks" in experiment
    assert "## Skill: boosting-demand-models" in experiment
    assert "XGBRegressor" in experiment
    assert "LightGBM native API" in experiment
    worktree = store.worktrees_dir / "loop0001"
    asyncio.run(_run("git", "worktree", "remove", "--force", str(worktree), cwd=repo))
