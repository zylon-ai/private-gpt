from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class SessionMount(ABC):
    """One mount point for a local session.

    prepare() is awaited concurrently at session creation and returns the real
    Path on the local filesystem. The PathTranslator is then built from the
    resolved paths.

    Future local-filesystem backends (s3fs, tmpfs, etc.) implement this interface
    without touching the session or provider code.
    """

    canonical: str
    writable: bool

    @abstractmethod
    async def prepare(self) -> Path:
        """Initialize the mount and return its real filesystem path."""

    async def teardown(self) -> None:  # noqa: B027
        """Optional cleanup. Default is a no-op."""


class LocalMount(SessionMount):
    """Simple writable-or-read-only local directory."""

    def __init__(self, canonical: str, path: Path, *, writable: bool) -> None:
        self.canonical = canonical
        self.writable = writable
        self._path = path

    async def prepare(self) -> Path:
        self._path.mkdir(parents=True, exist_ok=True)
        return self._path


class ReadOnlyMount(SessionMount):
    """Generic read-only mount: materialises pre-fetched file content to a local dir.

    Works for any ContentBundle — skills, plugins, or future content types.
    The cache directory is shared across sessions: if it already exists (from a
    previous session), files are not re-written.
    """

    writable: bool = False

    def __init__(
        self,
        canonical: str,
        files: dict[str, bytes],
        cache_dir: Path,
    ) -> None:
        self.canonical = canonical
        self.writable = False
        self._files = files
        self._cache_dir = cache_dir

    async def prepare(self) -> Path:
        if not self._cache_dir.exists():
            for rel_path, content in self._files.items():
                target = self._cache_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            for f in self._cache_dir.rglob("*"):
                if f.is_file():
                    f.chmod(0o444)
        return self._cache_dir
