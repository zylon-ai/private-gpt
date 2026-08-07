"""Namespace registry for the ZGPT filesystem platform.

A namespace maps a logical name (e.g. "session", "artifacts", "skills") to a
local filesystem root and a default access mode.  The registry is loaded from
settings at startup and fails fast when a configured root is non-empty but
missing or unreadable.  Namespaces whose root is empty/unset are skipped so
that optional namespaces (e.g. artifacts, skills) do not prevent startup in
local development environments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from injector import inject, singleton

from private_gpt.settings.settings import FilesystemsSettings, NamespaceConfig, Settings

logger = logging.getLogger(__name__)


@singleton
class NamespaceRegistry:
    """Resolves namespace names to their local root paths.

    Loaded from ``settings.filesystems.namespaces`` at injection time.
    Raises ``RuntimeError`` at startup if any configured (non-empty) root is
    missing or unreadable — intentional fail-fast behaviour.

    Namespaces with an empty root are silently skipped and will not be
    resolvable; attempting to look them up raises ``KeyError``.
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        fs_settings: FilesystemsSettings = settings.filesystems
        self._namespaces: dict[str, NamespaceConfig] = {}

        for name, cfg in fs_settings.namespaces.items():
            if not cfg.root or not cfg.root.strip():
                logger.debug("Namespace '%s': root not configured — skipping.", name)
                continue

            root = Path(cfg.root)
            if not root.exists():
                raise RuntimeError(
                    f"Namespace '{name}': configured root '{cfg.root}' does not exist. "
                    "Ensure the mount is available before starting."
                )
            if not os.access(str(root), os.R_OK):
                raise RuntimeError(
                    f"Namespace '{name}': configured root '{cfg.root}' is not readable."
                )
            self._namespaces[name] = cfg
            logger.info(
                "Namespace '%s' registered at '%s' (mode=%s)",
                name,
                cfg.root,
                cfg.default_mode,
            )

    def get(self, namespace: str) -> NamespaceConfig:
        """Return the config for *namespace*, raising ``KeyError`` if unknown."""
        if namespace not in self._namespaces:
            raise KeyError(
                f"Unknown namespace '{namespace}'. Available: {sorted(self._namespaces)}"
            )
        return self._namespaces[namespace]

    def root(self, namespace: str) -> Path:
        """Convenience accessor: return only the root path for *namespace*."""
        return Path(self.get(namespace).root)

    def all_names(self) -> list[str]:
        """Return all registered namespace names."""
        return list(self._namespaces)
