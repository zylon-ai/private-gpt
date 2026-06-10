from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from private_gpt.components.code_execution.content_bundle import BundledFile
    from private_gpt.components.sandbox.base import SandboxSession


class SandboxMountSpec(BaseModel):
    """Canonical mount point visible to the agent — backend-agnostic."""

    canonical: str  # e.g. "/home/agent/" — must end with "/"
    writable: bool = True


class LocalMountSpec(SandboxMountSpec):
    """Local-filesystem mount: adds the real path used by BashExecutorSandbox."""

    real_path: Path


class SessionMount(ABC):
    """Describes how to set up one mount point using sandbox APIs."""

    @property
    @abstractmethod
    def spec(self) -> SandboxMountSpec:
        ...

    @abstractmethod
    async def prepare(self, sandbox: SandboxSession) -> None:
        """Initialize this mount via sandbox APIs."""

    async def teardown(self, sandbox: SandboxSession) -> None:  # noqa: B027
        """Optional cleanup. Default is a no-op."""


class LocalMount(SessionMount):
    """Creates a directory via sandbox.make_dir()."""

    def __init__(self, spec: SandboxMountSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> SandboxMountSpec:
        return self._spec

    async def prepare(self, sandbox: SandboxSession) -> None:
        await sandbox.make_dir(self._spec.canonical)


class ReadOnlyMount(SessionMount):
    """Materialises BundledFiles via sandbox.initialize_mount(). Idempotent.

    For local backends path_exists() checks the real cache directory; if it
    already exists the files are not re-written.
    """

    def __init__(self, spec: SandboxMountSpec, files: list[BundledFile]) -> None:
        self._spec = spec
        self._files = files

    @property
    def spec(self) -> SandboxMountSpec:
        return self._spec

    async def prepare(self, sandbox: SandboxSession) -> None:
        if not await sandbox.path_exists(self._spec.canonical):
            await sandbox.initialize_mount(self._spec.canonical, self._files)
