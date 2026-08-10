import pytest


@pytest.fixture(autouse=True)
def _no_editor_popups(monkeypatch):
    """Never pop plan/findings files open in an editor from the test suite."""
    monkeypatch.setenv("AUTORESEARCH_NO_OPEN", "1")
