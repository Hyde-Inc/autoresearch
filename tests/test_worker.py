import asyncio
from pathlib import Path

from autoresearch.worker import _run, ensure_seed_repo


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
