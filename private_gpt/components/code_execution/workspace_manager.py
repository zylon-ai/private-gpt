from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class WorkspaceManager(ABC):
    """Manages the filesystem root for code execution session workspaces.

    Abstracts over local-filesystem and cloud-backed (e.g. s3fs) storage so
    that LocalCodeExecutionProvider and its subclasses are storage-agnostic.
    """

    @property
    @abstractmethod
    def sessions_path(self) -> Path:
        """Root directory under which per-session subdirs are created."""

    @abstractmethod
    def ensure_mounted(self) -> None:
        """Ensure the backing storage is mounted and ready. Idempotent."""

    @abstractmethod
    def unmount(self) -> None:
        """Release the backing storage mount (no-op for local)."""

    def prepare_session_dirs(self, session_id: str) -> dict[str, Path]:
        """Create the three canonical session dirs and return canonical→real_path map.

        The returned dict is always ordered:
          /home/agent/                → {session}/workspace
          /mnt/user-data/uploads/    → {session}/uploads
          /mnt/user-data/outputs/    → {session}/outputs
        """
        base = self.sessions_path / session_id
        paths: dict[str, Path] = {
            "/home/agent/": base / "workspace",
            "/mnt/user-data/uploads/": base / "uploads",
            "/mnt/user-data/outputs/": base / "outputs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        return paths


class LocalWorkspaceManager(WorkspaceManager):
    """Plain local filesystem — no cloud dependencies, just mkdir."""

    def __init__(self, base: Path) -> None:
        self._base = base

    @property
    def sessions_path(self) -> Path:
        return self._base

    def ensure_mounted(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)

    def unmount(self) -> None:
        pass
