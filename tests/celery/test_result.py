from unittest.mock import MagicMock

import pytest
from celery.exceptions import TimeoutError as CeleryTimeoutError

from private_gpt.celery.result import wait_for_celery_result


def test_wait_for_celery_result_polls_until_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = MagicMock(id="task-1")
    result.ready.side_effect = [False, True]
    result.failed.return_value = False
    result.result = {"result": "worker result"}
    sleep = MagicMock()
    monkeypatch.setattr("private_gpt.celery.result.time.sleep", sleep)

    response = wait_for_celery_result(result, timeout=42)

    assert response == {"result": "worker result"}
    assert result.ready.call_count == 2
    result.failed.assert_called_once_with()
    sleep.assert_called_once_with(0.1)


def test_wait_for_celery_result_raises_worker_exception() -> None:
    result = MagicMock(id="task-1")
    result.ready.return_value = True
    result.failed.return_value = True
    result.result = ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        wait_for_celery_result(result)


def test_wait_for_celery_result_enforces_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = MagicMock(id="task-1")
    result.ready.return_value = False
    monotonic = MagicMock(side_effect=[0.0, 1.0])
    monkeypatch.setattr("private_gpt.celery.result.time.monotonic", monotonic)

    with pytest.raises(CeleryTimeoutError, match="task-1"):
        wait_for_celery_result(result, timeout=0.5)
