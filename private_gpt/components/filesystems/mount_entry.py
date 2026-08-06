"""Mount entry models for the ZGPT filesystem platform (T4.2).

Mount entries are emitted by the Backend on each chat turn and resolve
to MountSpec objects that are bind-mounted into the container.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MountEntry(BaseModel):
    """A generic mount entry from the Backend's mount plan."""

    namespace: str = Field(
        description="Logical namespace (e.g. 'artifacts', 'session', 'skills')."
    )
    scope: str = Field(
        description="Opaque scope id within the namespace (e.g. org-id)."
    )
    path: str = Field(description="Relative path within the scope.")
    target: str = Field(
        description="Absolute container path where the file should be visible."
    )
    mode: Literal["rw", "ro"] = Field(default="rw", description="Access mode.")
    etag: str | None = Field(default=None, description="Optional content checksum.")
    artifact_id: str | None = Field(
        default=None, description="Source artifact id for correlation."
    )
