import asyncio
import shutil
from pathlib import Path

from autoresearch.worker import _run, ensure_seed_repo


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
