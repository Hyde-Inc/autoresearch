from pathlib import Path

from autoresearch.opencode import OpenCodeResult
from autoresearch.survey import cached_survey, survey_repo


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "project"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "arima.py").write_text("print('model')\n")
    (repo / "README.md").write_text("# Demo\n")
    (repo / ".autoresearch" / "task").mkdir(parents=True)
    (repo / ".autoresearch" / "task" / "secret.txt").write_text("holdout")
    return repo


def test_survey_runs_opencode_on_a_copy_and_caches(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    seen: dict = {}

    async def fake_opencode(cwd: Path, prompt: str, model: str, timeout_s: int, log_path: Path):
        seen["cwd"] = cwd
        seen["had_autoresearch"] = (cwd / ".autoresearch").exists()
        seen["had_model"] = (cwd / "models" / "arima.py").exists()
        # The sandbox must be a git repo so opencode anchors its project root
        # here instead of escaping to a parent repository.
        seen["had_git"] = (cwd / ".git").is_dir()
        (cwd / "SURVEY.md").write_text("# Repository survey\n\n## Models\nmodels/arima.py\n")
        return OpenCodeResult(0, "s1", None)

    monkeypatch.setattr("autoresearch.survey.run_opencode", fake_opencode)
    report = survey_repo(repo, "openrouter/moonshotai/kimi-k3")
    assert report is not None and "models/arima.py" in report
    # The agent worked on a throwaway copy, never the real repo...
    assert seen["cwd"] != repo
    assert seen["had_git"] is True
    assert seen["had_model"] is True
    # ...and the copy excludes .autoresearch (protected splits live there).
    assert seen["had_autoresearch"] is False
    assert not (repo / "SURVEY.md").exists()
    # Findings are cached inside the repo for later sessions.
    assert cached_survey(repo) == report


def test_survey_cache_short_circuits(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    cache = repo / ".autoresearch" / "survey.md"
    cache.write_text("cached findings")

    def explode(*args, **kwargs):
        raise AssertionError("opencode must not run when a cache exists")

    monkeypatch.setattr("autoresearch.survey.run_opencode", explode)
    assert survey_repo(repo, "m") == "cached findings"


def test_survey_falls_back_when_opencode_produces_nothing(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)

    async def silent(cwd: Path, prompt: str, model: str, timeout_s: int, log_path: Path):
        return OpenCodeResult(127, None, "opencode is not installed or not on PATH")

    monkeypatch.setattr("autoresearch.survey.run_opencode", silent)
    assert survey_repo(repo, "m") is None
    assert cached_survey(repo) is None
