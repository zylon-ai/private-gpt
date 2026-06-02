from private_gpt.components.sandbox.base import SandboxProvider, SandboxProviderFactory
from private_gpt.components.sandbox.local import LocalSandboxProvider
from private_gpt.settings.settings import Settings

_PROVIDERS: dict[str, SandboxProviderFactory] = {"local": LocalSandboxProvider}


def register_sandbox(name: str, provider: SandboxProviderFactory) -> None:
    _PROVIDERS[name] = provider


class SandboxProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, SandboxProvider] = {}

    def get_provider(self, name: str) -> SandboxProvider:
        provider = self._providers.get(name)
        if provider is not None:
            return provider

        provider_factory = _PROVIDERS.get(name)
        if provider_factory is None:
            available = ", ".join(sorted(_PROVIDERS)) or "none"
            raise ValueError(
                f"Sandbox provider '{name}' is not registered. Available: {available}"
            )
        provider = provider_factory(self._settings)
        self._providers[name] = provider
        return provider
