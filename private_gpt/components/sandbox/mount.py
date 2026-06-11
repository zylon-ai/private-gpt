from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class SandboxMountSpec(BaseModel):
    """Canonical mount point visible to the agent — backend-agnostic."""

    canonical: str  # e.g. "/home/agent/" — must end with "/"
    writable: bool = True


class LocalMountSpec(SandboxMountSpec):
    """Local-filesystem mount: adds the real path used by BashExecutorSandbox."""

    real_path: Path


class VolumeSpec(BaseModel):
    """Host directory bind-mounted into a sandbox at creation time."""

    name: str
    host_path: Path
    mount_path: str  # canonical path inside the sandbox, e.g. "/home/agent/"
    read_only: bool = False
