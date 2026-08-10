"""Input model for requesting content mounts in the chat API.

``MountEntry`` is an input model of the chat request body: API clients use
it to request that a file or folder be mounted into the sandbox where the
assistant runs during a chat turn.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MountEntry(BaseModel):
    """Declares a file or folder to mount into the sandbox.

    Part of the chat request body (``mounts`` field). Declares *what*
    content should be visible inside the sandbox and *where* it should
    appear. Content is located one of two ways:

    - ``uri`` (preferred): a content origin (``s3://...``, ``https://...``,
      ``data:...`` or a local disk path). The mount is lazy — content is
      fetched only when the backing folder is absent or empty.
    - ``namespace``/``scope``/``path``: a reference into a registered
      filesystem namespace, resolved to the exact host file/folder.
    """

    namespace: str = Field(
        description="Registered namespace name (e.g. 'session', 'skills', or a custom namespace)."
    )
    scope: str = Field(
        description="Opaque scope id within the namespace (e.g. thread-id)."
    )
    path: str = Field(description="Relative path within the scope.")
    target: str = Field(
        description="Absolute container path where the content should be visible."
    )
    mode: Literal["rw", "ro"] = Field(default="rw", description="Access mode.")
    etag: str | None = Field(default=None, description="Optional content checksum.")
    uri: str | None = Field(
        default=None,
        description=(
            "Content origin: s3://, https://, data: or a local disk path. "
            "When set, the mount is lazy — content is "
            "fetched only if the backing folder is absent or empty."
        ),
    )
