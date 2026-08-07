from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

AccessMode = Literal["rw", "ro"]


class MountFile(BaseModel):
    """A file that will exist inside a mounted folder.

    Produced by ``UriSource.fetch()`` when a lazy mount's folder must be
    materialized from its URI before the mount becomes usable.
    """

    path: str  # relative to the mount folder, e.g. "SKILL.md"
    content: bytes
    permissions: int = 0o444  # Unix permissions


def _fetch_from_uri(uri: str) -> Callable[[], Awaitable[list[MountFile]]]:
    """Build the default fetch for a URI-backed mount: load bytes from the URI."""

    async def fetch() -> list[MountFile]:
        import asyncio

        from private_gpt.server.ingest.uri_loader import load_file_from_uri

        binary = await asyncio.to_thread(load_file_from_uri, uri)
        filename = Path(urlparse(uri).path).name or "file"
        return [MountFile(path=filename, content=binary.read(), permissions=0o444)]

    return fetch


class UriSource(BaseModel):
    """Where a mount's content comes from (lazy materialization).

    A mount backed by a UriSource is *eager* when its host folder already has
    content (bind-mount directly — ``fetch`` is never called) and *lazy* when
    the folder is empty: then ``fetch()`` materializes the content from
    ``uri`` before the mount is usable.

    The URI may be anything ``load_file_from_uri`` understands — ``s3://``,
    ``https://``, ``data:`` or a local disk path — so a mount works the same
    in a remote network sandbox and in local private-gpt.
    """

    uri: str
    fetch: Callable[[], Awaitable[list[MountFile]]] = Field(exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _restore_fetch(cls, data: object) -> object:
        """Recreate the excluded fetch callable when a UriSource is rebuilt."""
        if (
            isinstance(data, dict)
            and "fetch" not in data
            and (uri_val := data.get("uri"))
        ):
            return {**data, "fetch": _fetch_from_uri(str(uri_val))}
        return data

    @classmethod
    def from_uri(cls, uri: str) -> UriSource:
        """Build a UriSource whose fetch loads bytes via the generic URI loader."""
        return cls(uri=uri, fetch=_fetch_from_uri(uri))


class Mount(BaseModel):
    """A directory visible inside a sandbox.

    - ``target``: where the directory appears in the sandbox (ends with "/").
    - ``access``: "rw" or "ro".
    - ``host_path``: the host directory backing the mount. When it already
      holds content the mount is eager (bind-mount it); when it is empty the
      mount is lazy and ``uri_source`` provides the content.
    - ``uri_source``: where to fetch the content from when ``host_path`` is
      absent or empty (any URI supported by ``load_file_from_uri``).
    - ``etag``: optional content checksum, used for change detection.
    """

    target: str  # e.g. "/home/agent/" — must end with "/"
    access: AccessMode = "ro"
    host_path: Path | None = None  # eager: folder already on the host
    uri_source: UriSource | None = None  # lazy: fetch from any URI
    name: str = ""
    description: str = ""
    etag: str | None = Field(
        default=None, description="Optional content checksum for change detection."
    )
