import io

from rich.console import Console

from autoresearch.slash import SessionSettings, handle_slash


def _console() -> Console:
    return Console(file=io.StringIO(), width=100)


def test_plain_text_is_not_handled() -> None:
    settings = SessionSettings()
    assert handle_slash("reduce wmape please", settings, _console()).handled is False


def test_goal_and_baseline_produce_notes() -> None:
    settings = SessionSettings()
    goal = handle_slash("/goal reduce wmape", settings, _console())
    baseline = handle_slash("/baseline models/arima.py", settings, _console())
    assert settings.goal == "reduce wmape"
    assert settings.baseline_path == "models/arima.py"
    assert "reduce wmape" in goal.note
    assert "models/arima.py" in baseline.note


def test_n_agents_bounds_and_bad_values() -> None:
    settings = SessionSettings()
    console = _console()
    assert handle_slash("/n_agents 4", settings, console).note == "n_agents set to 4"
    assert settings.n_agents == 4
    # Out of range and non-numeric are handled without changing settings.
    assert handle_slash("/n_agents 9", settings, console).note is None
    assert handle_slash("/n_agents lots", settings, console).note is None
    assert settings.n_agents == 4


def test_metric_guardrail_rounds_timeout() -> None:
    settings = SessionSettings()
    console = _console()
    handle_slash("/metric rmse", settings, console)
    handle_slash("/guardrail bias_pct within -8..8", settings, console)
    handle_slash("/rounds 2", settings, console)
    handle_slash("/timeout 300", settings, console)
    assert settings.metric == "rmse"
    assert "bias_pct within -8..8" in settings.guardrails
    assert settings.rounds == 2
    assert settings.timeout_s == 300
    assert handle_slash("/metric accuracy", settings, console).note is None
    assert settings.metric == "rmse"


def test_unknown_command_and_help_are_handled_locally() -> None:
    settings = SessionSettings()
    console = _console()
    assert handle_slash("/wat", settings, console).handled is True
    assert handle_slash("/help", settings, console).handled is True
    assert handle_slash("/status", settings, console).handled is True


def test_overrides_shape_matches_task_config() -> None:
    settings = SessionSettings(n_agents=2, metric="wmape", rounds=3, timeout_s=600)
    overrides = settings.overrides()
    assert overrides["agents"] == {"count": 2, "timeout_s": 600}
    assert overrides["budget"] == {"rounds": 3}
    assert overrides["metric"] == {"name": "wmape", "direction": "min"}
