import pytest

from private_gpt.worker import modes


def test_celery_mode_forwards_worker_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[list[str]]] = []
    monkeypatch.setenv("API_ENABLED", "false")
    monkeypatch.setattr(modes, "_run_processes", lambda value: commands.append(value))

    modes.run_celery(
        ["--without-gossip"],
        max_tasks_per_child_resolver=lambda: 1000,
    )

    assert commands[0][0][-1] == "--without-gossip"
