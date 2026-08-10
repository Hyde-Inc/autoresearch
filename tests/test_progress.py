from autoresearch.progress import Activity


class FakeStatus:
    def __init__(self, message: str) -> None:
        self.messages = [message]
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def update(self, message: str) -> None:
        self.messages.append(message)

    def stop(self) -> None:
        self.stopped = True


class FakeConsole:
    is_terminal = True

    def __init__(self) -> None:
        self.indicator: FakeStatus | None = None

    def status(self, message: str, spinner: str) -> FakeStatus:
        assert spinner == "dots"
        self.indicator = FakeStatus(message)
        return self.indicator


def test_activity_starts_updates_and_stops_with_factual_phase() -> None:
    console = FakeConsole()
    with Activity(console, "Running agents", interval_s=60, phrases=("cooking...",)) as activity:
        assert console.indicator is not None
        assert console.indicator.started
        assert "Running agents" in console.indicator.messages[0]
        assert "cooking..." in console.indicator.messages[0]
        activity.update("Reviewing results")
        assert "Reviewing results" in console.indicator.messages[-1]
    assert console.indicator.stopped


def test_activity_is_silent_without_a_terminal() -> None:
    console = FakeConsole()
    console.is_terminal = False
    with Activity(console, "Running agents") as activity:
        activity.update("Still running")
    assert console.indicator is None
