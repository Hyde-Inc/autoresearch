import asyncio
import shutil
from pathlib import Path

from autoresearch.worker import _run, build_worker_prompt, ensure_seed_repo


def test_seed_copy_supports_worktrees(tmp_path: Path) -> None:
    source = Path("examples/demand_forecasting/seed")
    repo = tmp_path / "repo"
    tree = tmp_path / "worktree"
    shutil.copytree(source, repo)
    asyncio.run(ensure_seed_repo(repo))
    code, output = asyncio.run(
        _run("git", "worktree", "add", "-b", "autoresearch/test", str(tree), "main", cwd=repo)
    )
    assert code == 0, output
    assert (tree / "solution/train.py").exists()
    asyncio.run(_run("git", "worktree", "remove", "--force", str(tree), cwd=repo))


def test_worker_prompt_contains_context_and_boundaries(tmp_path: Path) -> None:
    prompt = build_worker_prompt(
        tmp_path,
        "Forecast every requested SKU and date.",
        "# Experiment: XGBoost\n\nUse causal lag features.",
    )
    assert str(tmp_path) in prompt
    assert "Forecast every requested SKU and date." in prompt
    assert "Use causal lag features." in prompt
    assert "Do not search for TASK.md" in prompt
    assert "never inspect the parent directory" in prompt
