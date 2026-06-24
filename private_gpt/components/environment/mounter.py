from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from private_gpt.components.environment.layout import DEFAULT_SESSION_LAYOUT
from private_gpt.components.sandbox.content_bundle import StoredBundle
from private_gpt.components.sandbox.mount import SandboxMountSpec, VolumeSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from private_gpt.components.environment.layout import SessionMountDef
    from private_gpt.components.sandbox.base import SandboxSession
    from private_gpt.components.sandbox.content_bundle import ContentBundle


class Mounter(ABC):
    """Single owner of all mount logic for an environment.

    Decides which host directories back the session filesystem (if any),
    which mount specs the sandbox enforces writability with, and how the
    layout and content bundles are materialised inside the sandbox.
    """

    def __init__(
        self, layout: Sequence[SessionMountDef] = DEFAULT_SESSION_LAYOUT
    ) -> None:
        self._layout = tuple(layout)

    @property
    def layout(self) -> tuple[SessionMountDef, ...]:
        return self._layout

    @property
    def workspace_canonical(self) -> str:
        """The canonical working directory: first writable layout entry."""
        return next(m.canonical for m in self._layout if m.writable)

    def ensure_ready(self) -> None:  # noqa: B027 — optional hook, default no-op
        """One-time setup of the backing storage (e.g. mount s3fs). Idempotent."""

    @abstractmethod
    def session_volumes(self, session_id: str) -> list[VolumeSpec] | None:
        """Host volumes backing this session's layout dirs, or None when not host-backed.

        Only covers the fixed session layout (workspace, uploads, outputs).
        Bundle/skill content is always injected post-creation via prepare_bundles().
        Implementations create the host directories they return. Idempotent.
        """

    def mount_specs(
        self, extra_bundles: list[ContentBundle] | None = None
    ) -> list[SandboxMountSpec]:
        """Canonical mount specs the sandbox uses for writability enforcement."""
        specs: list[SandboxMountSpec] = [
            SandboxMountSpec(canonical=m.canonical, writable=m.writable)
            for m in self._layout
        ]
        specs.extend(
            SandboxMountSpec(canonical=b.canonical_path, writable=b.writable)
            for b in extra_bundles or []
        )
        return specs

    async def prepare(
        self,
        sandbox: SandboxSession,
        session_id: str,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> None:
        """Materialise the layout and bundles inside the sandbox. Idempotent."""
        if self.session_volumes(session_id) is None:
            await asyncio.gather(*[sandbox.make_dir(m.canonical) for m in self._layout])
        await self.prepare_bundles(sandbox, extra_bundles)

    async def prepare_bundles(
        self,
        sandbox: SandboxSession,
        extra_bundles: list[ContentBundle] | None = None,
    ) -> None:
        """Materialise bundles not already present (e.g. via a direct volume).

        Also runs on environment reuse, so content activated mid-session (a
        newly enabled skill) appears in the live sandbox via the copy fallback.
        """
        for bundle in extra_bundles or []:
            if await sandbox.path_exists(bundle.canonical_path):
                continue
            files = (
                await bundle.fetch()
                if isinstance(bundle, StoredBundle)
                else bundle.files
            )
            await sandbox.initialize_mount(bundle.canonical_path, files)


class SandboxDirMounter(Mounter):
    """No host backing: layout dirs are created inside the sandbox itself.

    Files live and die with the sandbox — for development or ephemeral use.
    """

    def session_volumes(self, session_id: str) -> list[VolumeSpec] | None:
        return None


class LocalDirMounter(Mounter):
    """Host-directory backing under a local base path.

    Layout dirs live under ``{base}/sessions/{session_id}/{name}`` and survive
    sandbox restarts; bundle files are cached under ``{base}/content_cache``.
    """

    def __init__(
        self,
        base: Path,
        layout: Sequence[SessionMountDef] = DEFAULT_SESSION_LAYOUT,
    ) -> None:
        super().__init__(layout)
        self._sessions = base / "sessions"

    def ensure_ready(self) -> None:
        self._sessions.mkdir(parents=True, exist_ok=True)

    def session_volumes(self, session_id: str) -> list[VolumeSpec] | None:
        base = self._sessions / session_id
        volumes: list[VolumeSpec] = []
        for mount in self._layout:
            host = base / mount.name
            host.mkdir(parents=True, exist_ok=True)
            volumes.append(
                VolumeSpec(
                    name=mount.name,
                    host_path=host,
                    mount_path=mount.canonical,
                    read_only=not mount.writable,
                )
            )
        return volumes
