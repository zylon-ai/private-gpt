from injector import inject, singleton

from private_gpt.components.sandbox.base import SandboxSession
from private_gpt.components.sandbox.registry import SandboxProviderRegistry
from private_gpt.settings.settings import Settings


@singleton
class SandboxComponent:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = SandboxProviderRegistry(settings)

    def create_session(
        self, user_id: str | None = None, timeout: int | None = None
    ) -> SandboxSession | None:
        provider_name = self._settings.sandbox.provider
        if provider_name is None:
            return None
        provider = self._registry.get_provider(provider_name)
        return provider.create_session(user_id=user_id, timeout=timeout)
