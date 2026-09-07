from unittest.mock import MagicMock

import pytest

from private_gpt.arq.runner import _keep_result_seconds, _queue_name, _task_packages


def test_task_packages_are_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PGPT_ARQ_TASK_PACKAGES",
        "private_gpt.arq.tasks.chat, custom.tasks ",
    )

    assert _task_packages() == (
        "private_gpt.arq.tasks.chat",
        "custom.tasks",
    )


def test_task_packages_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGPT_ARQ_TASK_PACKAGES", raising=False)

    with pytest.raises(ValueError, match="PGPT_ARQ_TASK_PACKAGES"):
        _task_packages()


def test_queue_is_loaded_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPT_ARQ_QUEUE", "chat")

    assert _queue_name() == "private_gpt:arq:queue:chat"


def test_queue_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGPT_ARQ_QUEUE", raising=False)

    with pytest.raises(ValueError, match="PGPT_ARQ_QUEUE"):
        _queue_name()


def test_keep_result_outlives_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGPT_ARQ_KEEP_RESULT", raising=False)
    current_settings = MagicMock()
    current_settings.scheduler.chat.callback_timeout_seconds = 120

    assert _keep_result_seconds(current_settings) == 420


def test_keep_result_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPT_ARQ_KEEP_RESULT", "900")

    assert _keep_result_seconds(MagicMock()) == 900
