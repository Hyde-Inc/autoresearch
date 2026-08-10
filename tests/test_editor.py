from pathlib import Path

from autoresearch import editor


def test_env_guard_disables_opening(tmp_path: Path) -> None:
    # AUTORESEARCH_NO_OPEN is set for the whole suite by conftest.
    assert editor.open_in_editor(tmp_path / "plan.md") is False


def test_opens_with_first_available_cli(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AUTORESEARCH_NO_OPEN")
    launched: list[list[str]] = []
    monkeypatch.setattr(
        editor.shutil, "which", lambda name: "/bin/cursor" if name == "cursor" else None
    )
    monkeypatch.setattr(editor.subprocess, "Popen", lambda cmd, **kw: launched.append(cmd))
    target = tmp_path / "plan.md"
    assert editor.open_in_editor(target) is True
    assert launched == [["/bin/cursor", "--reuse-window", str(target)]]
