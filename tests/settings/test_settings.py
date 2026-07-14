import pytest

from private_gpt.settings.settings import Settings, settings
from private_gpt.settings.settings_loader import merge_settings
from tests.fixtures.mock_injector import MockInjector, unsafe_settings


def test_settings_are_loaded_and_merged() -> None:
    assert settings().server.env_name == "test"


def test_settings_can_be_overriden(injector: MockInjector) -> None:
    injector.bind_settings({"server": {"env_name": "overriden"}})
    mocked_settings = injector.get(Settings)
    assert mocked_settings.server.env_name == "overriden"


def test_chat_scheduler_rejects_unsupported_celery_mode() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "scheduler": {"chat": {"mode": "celery"}},
                "stream": {"broker": "redis"},
                "celery": {"use_workers": True},
            },
        ]
    )

    with pytest.raises(ValueError, match=r"Unsupported scheduler\.chat\.mode='celery'"):
        Settings(**merged)


def test_arq_chat_scheduler_requires_redis_stream_backend() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "scheduler": {"chat": {"mode": "arq"}},
                "stream": {"broker": "memory"},
            },
        ]
    )

    with pytest.raises(ValueError, match=r"stream\.broker=redis"):
        Settings(**merged)


@pytest.mark.parametrize("tool_mode", ["arq", "celery"])
def test_remote_tool_scheduler_requires_arq_chat(tool_mode: str) -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "scheduler": {
                    "chat": {"mode": "local"},
                    "tools": {"mode": tool_mode},
                },
            },
        ]
    )

    with pytest.raises(ValueError, match=r"scheduler\.chat\.mode='arq'"):
        Settings(**merged)
