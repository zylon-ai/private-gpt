from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

AccessMode = Literal["rw", "ro"]


class MountSpec(BaseModel):
    """A single mount: a canonical path visible to the agent, backed by a host path.

    This is the single mount model used across the platform. It replaces the
    three competing models that used to exist (``SandboxMountSpec``,
    ``LocalMountSpec`` and ``VolumeSpec``), which were the same four fields
    expressed with different names. The legacy names are still accepted on
    construction (``mount_path``/``canonical``, ``real_path``/``host_path``,
    ``writable``/``read_only``) and exposed as read-only properties so
    existing callers keep working.
    """

    canonical: str  # e.g. "/home/agent/" — must end with "/"
    host_path: Path | None = None
    read_only: bool = False
    name: str = ""
    description: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def access(self) -> AccessMode:
        return "ro" if self.read_only else "rw"

    @property
    def writable(self) -> bool:
        """Read-only inverse of ``read_only`` (legacy ``SandboxMountSpec`` API)."""
        return not self.read_only

    @property
    def mount_path(self) -> str:
        """Legacy ``VolumeSpec`` alias for ``canonical``."""
        return self.canonical

    @property
    def real_path(self) -> Path | None:
        """Legacy ``LocalMountSpec`` alias for ``host_path``."""
        return self.host_path

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: object) -> object:
        """Accept the legacy field names used by the old mount models."""
        if isinstance(data, dict):
            data = dict(data)
            if "mount_path" in data and "canonical" not in data:
                data["canonical"] = data.pop("mount_path")
            if "real_path" in data and "host_path" not in data:
                data["host_path"] = data.pop("real_path")
            if "writable" in data:
                data["read_only"] = not data.pop("writable")
        return data


# Backward-compatible aliases — all legacy names point at the one model.
VolumeSpec = MountSpec
SandboxMountSpec = MountSpec
