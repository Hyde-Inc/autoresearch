import pytest

from autoresearch import doctor


def test_check_tool_missing_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    check = doctor._check_tool("definitely-not-a-real-tool", "do a thing")
    assert check.status == "fail"
    assert "not on PATH" in check.detail


def test_check_tool_present_reports_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(doctor, "_tool_version", lambda _: "git version 2.39.5")
    check = doctor._check_tool("git", "create worktrees")
    assert check.status == "ok"
    assert "git version 2.39.5" in check.detail


def test_openrouter_missing_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    check = doctor._check_openrouter(check_api=True)
    assert check.status == "fail"


def test_openrouter_key_set_skips_live_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    check = doctor._check_openrouter(check_api=False)
    assert check.status == "ok"
    assert "skipped" in check.detail


def test_foundry_partial_credentials_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.example.com")
    monkeypatch.delenv("FOUNDRY_TOKEN", raising=False)
    check = doctor._check_foundry()
    assert check.status == "warn"
    assert "FOUNDRY_TOKEN" in check.detail


def test_run_checks_returns_all_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    names = [check.name for check in doctor.run_checks(check_api=False)]
    assert names == ["python", "git", "uv", "opencode", ".env", "openrouter", "foundry"]
