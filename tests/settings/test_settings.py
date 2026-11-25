from pathlib import Path

import pytest

from private_gpt.settings import settings_loader
from private_gpt.settings.settings import Settings, settings
from private_gpt.settings.settings_loader import merge_settings
from tests.fixtures.mock_injector import MockInjector, unsafe_settings


def test_settings_are_loaded_and_merged() -> None:
    assert settings().server.env_name == "test"


def test_settings_can_be_overriden(injector: MockInjector) -> None:
    injector.bind_settings({"server": {"env_name": "overriden"}})
    mocked_settings = injector.get(Settings)
    assert mocked_settings.server.env_name == "overriden"


@pytest.mark.parametrize(
    ("profiles", "overrides", "expected"),
    [
        ([], {}, (True, "parallel", "simple_text")),
        ([], {"PGPT_WEBSEARCH_ENABLED": "false"}, (False, "parallel", "simple_text")),
        ([], {"PGPT_WEBSEARCH_ENABLED": "true"}, (True, "parallel", "simple_text")),
        ([], {"PGPT_WEBSEARCH_PROVIDER": "brave"}, (False, "brave", "best_links")),
        (
            [],
            {"PGPT_WEBSEARCH_PROCESSOR": "clean_content"},
            (True, "parallel", "clean_content"),
        ),
        ([], {"PGPT_OFFLINE_MODE": "true"}, (False, "mock", "simple_text")),
        (["mock"], {}, (False, "mock", "best_links")),
        (["test"], {}, (False, "mock", "best_links")),
        (
            ["mock"],
            {"PGPT_WEBSEARCH_ENABLED": "true", "PGPT_WEBSEARCH_PROVIDER": "brave"},
            (True, "brave", "best_links"),
        ),
    ],
)
def test_shipped_web_search_defaults_respect_profiles_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    profiles: list[str],
    overrides: dict[str, str],
    expected: tuple[bool, str, str],
) -> None:
    monkeypatch.setattr(settings_loader, "active_profiles", ["default", *profiles])
    for variable in (
        "PGPT_WEBSEARCH_ENABLED",
        "PGPT_WEBSEARCH_PROVIDER",
        "PGPT_WEBSEARCH_PROCESSOR",
        "PGPT_OFFLINE_MODE",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(variable, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    web = Settings.model_validate(settings_loader.load_active_settings()).web_search
    assert (web.enabled, web.provider, web.processor) == expected


def test_web_search_profile_overrides_base_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "settings-deployment.yaml").write_text(
        "web_search:\n  enabled: false\n  provider: mock\n  processor: simple_text\n"
    )
    monkeypatch.setattr(settings_loader, "active_profiles", ["default", "deployment"])
    monkeypatch.setattr(
        settings_loader,
        "_settings_folders",
        [str(tmp_path), *settings_loader._settings_folders],
    )
    monkeypatch.setenv("PGPT_WEBSEARCH_ENABLED", "true")
    monkeypatch.setenv("PGPT_WEBSEARCH_PROVIDER", "brave")

    web = Settings.model_validate(settings_loader.load_active_settings()).web_search
    assert (web.enabled, web.provider, web.processor) == (False, "mock", "simple_text")


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


def test_celery_tool_scheduler_requires_arq_chat() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {
                "scheduler": {
                    "chat": {"mode": "local"},
                    "tools": {"mode": "celery"},
                },
            },
        ]
    )

    with pytest.raises(ValueError, match=r"scheduler\.chat\.mode='arq'"):
        Settings(**merged)


def test_tool_scheduler_rejects_unsupported_arq_mode() -> None:
    merged = merge_settings(
        [
            unsafe_settings,
            {"scheduler": {"tools": {"mode": "arq"}}},
        ]
    )

    with pytest.raises(ValueError, match=r"Unsupported scheduler\.tools\.mode='arq'"):
        Settings(**merged)
