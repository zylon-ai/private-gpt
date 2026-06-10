from injector import inject, singleton

from private_gpt.components.sandbox.base import (
    AsyncSandboxProvider,
    AsyncSandboxSession,
    SandboxProvider,
    SandboxSession,
)
from private_gpt.components.sandbox.mount import SandboxMountSpec
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
        assert isinstance(provider, SandboxProvider)

        return provider.create_session(user_id=user_id, timeout=timeout)

    async def acreate_session(
        self,
        user_id: str | None = None,
        timeout: int | None = None,
        bundle_specs: list[SandboxMountSpec] | None = None,
    ) -> AsyncSandboxSession | None:
        provider_name = self._settings.sandbox.provider
        if provider_name is None:
            return None

        provider = self._registry.get_provider(provider_name)
        assert isinstance(provider, AsyncSandboxProvider)

        return await provider.create_session(
            user_id=user_id, timeout=timeout, bundle_specs=bundle_specs
        )
