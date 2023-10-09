from private_gpt.settings.settings import settings


def test_settings_are_loaded_and_merged() -> None:
    assert settings.server.env_name == "test"
