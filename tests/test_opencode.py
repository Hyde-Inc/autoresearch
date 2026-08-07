from pathlib import Path

from autoresearch.opencode import build_opencode_command


def test_command_pins_agent_to_worktree() -> None:
    workspace = Path("/tmp/run/worktrees/agent-1")
    command = build_opencode_command(
        "/usr/local/bin/opencode",
        workspace,
        "Implement the experiment",
        "openrouter/example/model",
    )
    directory_index = command.index("--dir")
    assert command[directory_index + 1] == str(workspace)
    assert command[-1] == "Implement the experiment"
