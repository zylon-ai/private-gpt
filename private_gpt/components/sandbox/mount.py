from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from private_gpt.components.sandbox.content_bundle import BundledFile

AccessMode = Literal["rw", "ro"]


class StorageRef(BaseModel):
    """Reference to content in object storage, resolved lazily on demand.

    A mount backed by a storage ref is bind-mounted directly when the host
    folder is already present (infra volume / s3fs / local storage); the
    ``fetch`` callable is the hydration fallback used only when the folder
    is absent or empty.
    """

    prefix: str  # e.g. "skills/org-1/pdf/v1"
    fetch: Callable[[], Awaitable[list[BundledFile]]] = Field(exclude=True)


class MountSpec(BaseModel):
    """A single mount: a directory visible to the agent, backed by a host path.

    This is the single mount model used across the platform. It replaces the
    three competing models that used to exist (``SandboxMountSpec``,
    ``LocalMountSpec`` and ``VolumeSpec``), which were the same fields
    expressed with different names.

    A mount is always a directory — never a single file. ``target`` is the
    canonical path inside the sandbox (must end with "/"), ``source`` is the
    host directory backing it (None until resolved at container creation),
    and ``storage`` is the optional object-storage reference used to hydrate
    the source folder when it is not already materialized.

    Legacy names are still accepted on construction (``mount_path``/``canonical``,
    ``real_path``/``host_path``, ``writable``/``read_only``) and exposed as
    read-only properties so existing callers keep working.
    """

    target: str  # e.g. "/home/agent/" — must end with "/"
    access: AccessMode = "ro"  # replaces writable / read_only / mode
    source: Path | None = None  # host dir; None until resolved at creation
    name: str = ""
    description: str = ""
    etag: str | None = Field(
        default=None, description="Optional content checksum for change detection."
    )
    storage: StorageRef | None = Field(
        default=None,
        exclude=True,
        description=(
            "Object-storage reference used to hydrate the source folder when "
            "it is absent or empty. When the folder is already present the "
            "mount is wired directly with no fetch."
        ),
    )

    # --- Legacy aliases ----------------------------------------------------
    @property
    def canonical(self) -> str:
        """Legacy ``SandboxMountSpec`` alias for ``target``."""
        return self.target

    @property
    def mount_path(self) -> str:
        """Legacy ``VolumeSpec`` alias for ``target``."""
        return self.target

    @property
    def real_path(self) -> Path | None:
        """Legacy ``LocalMountSpec`` alias for ``source``."""
        return self.source

    @property
    def host_path(self) -> Path | None:
        """Legacy ``VolumeSpec`` alias for ``source``."""
        return self.source

    @property
    def read_only(self) -> bool:
        """Legacy ``VolumeSpec`` inverse of ``access``."""
        return self.access == "ro"

    @property
    def writable(self) -> bool:
        """Legacy ``SandboxMountSpec`` API — inverse of ``read_only``."""
        return self.access == "rw"

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: object) -> object:
        """Accept the legacy field names used by the old mount models."""
        if isinstance(data, dict):
            data = dict(data)
            if "mount_path" in data and "target" not in data:
                data["target"] = data.pop("mount_path")
            elif "canonical" in data and "target" not in data:
                data["target"] = data.pop("canonical")
            if "real_path" in data and "source" not in data:
                data["source"] = data.pop("real_path")
            elif "host_path" in data and "source" not in data:
                data["source"] = data.pop("host_path")
            if "writable" in data:
                data["access"] = "rw" if data.pop("writable") else "ro"
            elif "read_only" in data and "access" not in data:
                data["access"] = "ro" if data.pop("read_only") else "rw"
        return data


# Backward-compatible aliases — all legacy names point at the one model.
VolumeSpec = MountSpec
SandboxMountSpec = MountSpec
Mount = MountSpec
