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


def test_chat_scheduler_requires_redis_stream_backend() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "scheduler": {"chat": {"mode": "celery"}},
                "stream": {"broker": "memory"},
                "celery": {"use_workers": True},
            },
        ]
    )

    with pytest.raises(ValueError, match=r"stream\.broker=redis"):
        Settings(**merged)


def test_chat_scheduler_celery_mode_requires_redis_broker() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "scheduler": {"chat": {"mode": "celery"}},
                "stream": {"broker": "memory"},
            },
        ]
    )

    with pytest.raises(ValueError, match=r"stream\.broker=redis"):
        Settings(**merged)
