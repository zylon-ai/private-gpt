import pytest

from private_gpt.settings.settings import CelerySettings
from private_gpt.worker import modes


def test_celery_mode_forwards_worker_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[list[str]]] = []
    monkeypatch.setenv("API_ENABLED", "false")
    monkeypatch.setattr(modes, "_run_processes", lambda value: commands.append(value))

    modes.run_celery(
        ["--without-gossip"],
        celery_settings_provider=lambda: CelerySettings(),
    )

    assert commands[0][0][-1] == "--without-gossip"


def test_stateful_celery_worker_sets_memory_recycling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_STATEFUL_WORKER_TYPE", "tools")

    args = modes._build_celery_args(
        CelerySettings(
            max_tasks_per_child=100,
            max_memory_per_child=2_097_152,
        )
    )

    assert "--max-tasks-per-child=100" in args
    assert "--max-memory-per-child=2097152" in args
