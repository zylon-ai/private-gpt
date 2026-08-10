from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from private_gpt.components.environment.environment import Environment
    from private_gpt.components.environment.manager import EnvironmentManager
    from private_gpt.components.sandbox.mount import Mount, MountFile
    from private_gpt.settings.settings import NamespaceConfig

logger = logging.getLogger(__name__)

_LEDGER_DIR = ".hydration"


class HydratingEnvironmentManager:
    """Transparent wrapper that (re)hydrates namespace content before acquire.

    Sandboxes only ever see bind volumes: the host must already hold the
    content. In production a FUSE/s3fs mount provides it. In local development
    the volume roots are plain folders, so this wrapper syncs the host paths
    from each mount's URI first — gated per namespace by
    ``NamespaceConfig.hydration`` (default off), with an etag ledger to avoid
    re-reading/writing content that did not change.

    With no hydrating namespaces configured this wrapper is a pure pass-through
    and adds no I/O.
    """

    def __init__(
        self,
        manager: EnvironmentManager,
        namespaces: dict[str, NamespaceConfig],
    ) -> None:
        self._manager = manager
        self._hydrating: dict[str, NamespaceConfig] = {
            name: cfg for name, cfg in namespaces.items() if cfg.hydration
        }

    async def acquire(
        self,
        session_id: str,
        mounts: list[Mount] | None = None,
        sandbox_env: dict[str, str] | None = None,
    ) -> Environment:
        if mounts and self._hydrating:
            mounts = await self._hydrate(mounts)
        return await self._manager.acquire(session_id, mounts, sandbox_env)

    def release(self, session_id: str) -> None:
        self._manager.release(session_id)

    # ------------------------------------------------------------------
    # Hydration internals
    # ------------------------------------------------------------------

    async def _hydrate(self, mounts: list[Mount]) -> list[Mount]:
        hydrated: list[Mount] = []
        for mount in mounts:
            namespace = mount.source.namespace if mount.source else None
            config = self._hydrating.get(namespace or "")
            if config is None or mount.host_path is None or mount.uri_source is None:
                hydrated.append(mount)
                continue
            if await self._is_fresh(config, mount):
                hydrated.append(mount)
                continue
            if await self._materialize(config, mount):
                hydrated.append(mount)
            else:
                logger.warning(
                    "Dropping mount %s -> %s: could not hydrate from URI",
                    mount.host_path,
                    mount.target,
                )
        return hydrated

    async def _is_fresh(self, config: NamespaceConfig, mount: Mount) -> bool:
        assert mount.host_path is not None
        ledger = self._ledger_path(config, mount)
        if not ledger.exists():
            return False
        try:
            stored = json.loads(ledger.read_text())
        except (OSError, ValueError):
            return False
        if stored.get("etag") != mount.etag:
            return False
        if mount.is_folder:
            return mount.host_path.is_dir() and any(mount.host_path.iterdir())
        return mount.host_path.is_file()

    async def _materialize(self, config: NamespaceConfig, mount: Mount) -> bool:
        assert mount.host_path is not None
        assert mount.uri_source is not None
        files = await mount.uri_source.fetch()
        if not files:
            return False

        host = mount.host_path
        if mount.is_folder:
            await asyncio.to_thread(_write_folder, host, files)
        else:
            # A file mount binds one exact host file; fetch() returns the
            # content of that file (path/filename may differ from the mount).
            # Permissions follow the mount access so rw mounts stay writable
            # through the bind (and re-hydratable).
            permissions = 0o644 if mount.access == "rw" else 0o444
            await asyncio.to_thread(_write_file, host, files[0].content, permissions)

        ledger = self._ledger_path(config, mount)
        await asyncio.to_thread(_write_json, ledger, {"etag": mount.etag})
        logger.debug(
            "Hydrated %s -> %s (etag=%s)",
            mount.uri_source.uri,
            host,
            mount.etag,
        )
        return True

    def _ledger_path(self, config: NamespaceConfig, mount: Mount) -> Path:
        source = mount.source
        parts = [
            p
            for p in (source.scope if source else "", source.path if source else "")
            if p
        ]
        rel = "/".join(parts) if parts else "_"
        return Path(config.root) / _LEDGER_DIR / f"{rel}.json"


def _write_file(path: Path, content: bytes, permissions: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(permissions)


def _write_folder(host: Path, files: list[MountFile]) -> None:
    for f in files:
        relative = PurePosixPath(f.path)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        dest = host / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.content)
        dest.chmod(f.permissions)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
