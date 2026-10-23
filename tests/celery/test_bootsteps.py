from unittest.mock import Mock, call

import pytest

from private_gpt.celery import bootsteps


def test_worker_ready_does_not_warm_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warm = Mock()
    readiness_file = Mock()
    monkeypatch.setenv("PGPT_STATEFUL_WORKER_TYPE", "tools")
    monkeypatch.setattr(bootsteps, "_warm_stateful_worker", warm)
    monkeypatch.setattr(bootsteps, "READINESS_FILE", readiness_file)

    bootsteps.handle_worker_ready()

    warm.assert_not_called()
    readiness_file.touch.assert_called_once_with()


def test_worker_process_init_resets_before_warming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = Mock()
    monkeypatch.setenv("PGPT_STATEFUL_WORKER_TYPE", "tools")
    monkeypatch.setattr(
        "private_gpt.celery.base.StatefulBackgroundTask.reset_after_fork",
        lambda: operations("reset"),
    )
    monkeypatch.setattr(
        bootsteps,
        "_warm_stateful_worker",
        lambda: operations("warm"),
    )

    bootsteps.handle_worker_process_init()

    assert operations.call_args_list == [call("reset"), call("warm")]
