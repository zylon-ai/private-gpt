"""Mount entry models for the ZGPT filesystem platform.

Mount entries are emitted by the Backend on each chat turn and resolve
to Mount objects that are bind-mounted into the container.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MountEntry(BaseModel):
    """Backend request to make content visible inside a sandbox.

    The Backend declares *what* to mount and *where* it should appear inside
    the sandbox. Content is located one of two ways:

    - ``uri`` (preferred): a content origin understood by ``load_file_from_uri``
      (``s3://...``, ``https://...``, ``data:...`` or a local disk path). The
      mount is lazy — content is fetched only when the backing folder is empty.
    - ``namespace``/``scope``/``path``: a reference into the platform
      filesystem, resolved by ``MountRefResolver`` to a local host folder
      (eager when the folder already has content).
    """

    namespace: str = Field(
        description="Logical namespace (e.g. 'artifacts', 'session', 'skills')."
    )
    scope: str = Field(
        description="Opaque scope id within the namespace (e.g. org-id)."
    )
    path: str = Field(description="Relative path within the scope.")
    target: str = Field(
        description="Absolute container path where the content should be visible."
    )
    mode: Literal["rw", "ro"] = Field(default="rw", description="Access mode.")
    etag: str | None = Field(default=None, description="Optional content checksum.")
    artifact_id: str | None = Field(
        default=None, description="Source artifact id for correlation."
    )
    uri: str | None = Field(
        default=None,
        description=(
            "Content origin: s3://, https://, data: or disk path understood by "
            "load_file_from_uri. When set, the mount is lazy — content is "
            "fetched only if the backing folder is absent or empty."
        ),
    )
