from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from private_gpt.components.environment.layout import DEFAULT_SESSION_LAYOUT
from private_gpt.components.sandbox.mount import SandboxMountSpec, VolumeSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from private_gpt.components.environment.layout import SessionMountDef


class LayoutMounter(ABC):
    """Owns the session filesystem layout — workspace, uploads, outputs.

    Responsible only for structural concerns: which canonical paths make up
    the session's persistent directory tree and how they are backed on the
    host. Bundle content (skills, tools, ...) is a separate concern handled
    by ContentMounter implementations.
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
        """Canonical working directory: the first writable layout entry."""
        return next(m.canonical for m in self._layout if m.writable)

    def ensure_ready(self) -> None:  # noqa: B027 — optional hook, default no-op
        """One-time idempotent setup of backing storage (e.g. mount s3fs)."""

    @abstractmethod
    def session_volumes(self, session_id: str) -> list[VolumeSpec] | None:
        """Host volumes backing this session's layout dirs, or None if not host-backed.

        Only covers the fixed session layout (workspace, uploads, outputs).
        Bundle/skill volumes are declared by ContentMounter.prepare_volume().
        Implementations create the host directories they return. Idempotent.
        """

    def mount_specs(self) -> list[SandboxMountSpec]:
        """Canonical mount specs for the session layout (writability enforcement).

        Bundle specs are added separately by the EnvironmentManager so this
        class stays unaware of content.
        """
        return [
            SandboxMountSpec(canonical=m.canonical, writable=m.writable)
            for m in self._layout
        ]


# Backward-compatible alias so existing imports of `Mounter` keep working.
Mounter = LayoutMounter


class SandboxDirMounter(LayoutMounter):
    """No host backing — layout dirs are created inside the sandbox itself.

    Files live and die with the sandbox. Suitable for development or
    ephemeral use where persistence is not required.
    """

    def session_volumes(self, session_id: str) -> list[VolumeSpec] | None:
        return None


class LocalDirMounter(LayoutMounter):
    """Host-directory backing under a local base path.

    Layout dirs live under ``{base}/{name}/{session_id}`` so that each folder
    type sits at a top-level prefix — enabling per-folder MinIO lifecycle rules.
    Bundle content is handled by LocalStorageContentMounter or
    FetchContentMounter, not here.
    """

    def __init__(
        self,
        base: Path,
        layout: Sequence[SessionMountDef] = DEFAULT_SESSION_LAYOUT,
    ) -> None:
        super().__init__(layout)
        self._base = base

    def ensure_ready(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)

    def uploads_path(self, session_id: str) -> Path:
        return self._base / "uploads" / session_id

    def outputs_path(self, session_id: str) -> Path:
        return self._base / "outputs" / session_id

    def session_volumes(self, session_id: str) -> list[VolumeSpec] | None:
        volumes: list[VolumeSpec] = []
        for mount in self._layout:
            host = self._base / mount.name / session_id
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
