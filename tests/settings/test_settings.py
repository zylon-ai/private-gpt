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


def test_chat_worker_requires_redis_stream_backend() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "chat": {"use_chat_worker": True},
                "stream": {"broker": "memory"},
                "celery": {"use_workers": True},
            },
        ]
    )

    with pytest.raises(ValueError, match=r"stream\.broker=redis"):
        Settings(**merged)


def test_chat_worker_requires_real_celery_worker() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "chat": {"use_chat_worker": True},
                "stream": {"broker": "redis"},
                "celery": {"use_workers": False},
            },
        ]
    )

    with pytest.raises(ValueError, match=r"celery\.use_workers=true"):
        Settings(**merged)
