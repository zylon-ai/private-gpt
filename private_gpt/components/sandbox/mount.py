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


def _fetch_from_uri(
    uri: str, filename: str | None = None
) -> Callable[[], Awaitable[list[MountFile]]]:
    """Build the default fetch for a URI-backed file or directory mount."""

    async def fetch() -> list[MountFile]:
        import asyncio

        from private_gpt.server.ingest.uri_loader import load_file_from_uri

        local_path = (
            Path(urlparse(uri).path) if uri.startswith("file://") else Path(uri)
        )
        if local_path.is_dir():

            def read_directory() -> list[MountFile]:
                files: list[MountFile] = []
                for path in sorted(
                    path for path in local_path.rglob("*") if path.is_file()
                ):
                    files.append(
                        MountFile(
                            path=path.relative_to(local_path).as_posix(),
                            content=path.read_bytes(),
                            permissions=0o444,
                        )
                    )
                return files

            return await asyncio.to_thread(read_directory)

        binary = await asyncio.to_thread(load_file_from_uri, uri)
        resolved_filename = filename or Path(urlparse(uri).path).name or "file"
        return [
            MountFile(path=resolved_filename, content=binary.read(), permissions=0o444)
        ]

    return fetch


def _fetch_from_uris(
    sources: list[tuple[str, str | None]],
) -> Callable[[], Awaitable[list[MountFile]]]:
    """Build a fetcher that materializes several URI-backed files."""

    async def fetch() -> list[MountFile]:
        files: list[MountFile] = []
        for uri, filename in sources:
            files.extend(await _fetch_from_uri(uri, filename)())
        return files

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
    filename: str | None = Field(
        default=None,
        description="Optional filename to use when materializing the URI.",
    )
    additional_sources: list[tuple[str, str | None]] = Field(
        default_factory=list,
        description="Additional URI-backed files materialized in the same folder.",
    )
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
            filename_val = data.get("filename")
            filename = filename_val if isinstance(filename_val, str) else None
            additional_sources = _normalize_additional_sources(
                data.get("additional_sources")
            )
            sources = [(str(uri_val), filename), *additional_sources]
            return {
                **data,
                "fetch": _fetch_from_uris(sources),
            }
        return data

    @classmethod
    def from_uri(
        cls,
        uri: str,
        filename: str | None = None,
        additional_sources: list[tuple[str, str | None]] | None = None,
    ) -> UriSource:
        """Build a UriSource whose fetch loads bytes via the generic URI loader."""
        sources = additional_sources or []
        return cls(
            uri=uri,
            filename=filename,
            additional_sources=sources,
            fetch=_fetch_from_uris([(uri, filename), *sources]),
        )

    @property
    def sources(self) -> list[tuple[str, str | None]]:
        """Return every URI and target filename materialized by this source."""
        return [(self.uri, self.filename), *self.additional_sources]

    @property
    def is_composite(self) -> bool:
        return bool(self.additional_sources)

    @property
    def cache_key(self) -> tuple[tuple[str, str | None], ...]:
        """Stable source identity used when deciding whether mounts changed."""
        return tuple(self.sources)


def _normalize_additional_sources(
    sources: object,
) -> list[tuple[str, str | None]]:
    if not isinstance(sources, list):
        return []
    normalized: list[tuple[str, str | None]] = []
    for source in sources:
        if not isinstance(source, (list, tuple)) or len(source) != 2:
            continue
        uri, filename = source
        if not isinstance(uri, str):
            continue
        normalized.append((uri, filename if isinstance(filename, str) else None))
    return normalized


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
    # Generic source identity preserved from a Backend MountEntry.  These
    # fields are intentionally storage-neutral; providers may use them to
    # resolve a namespace-backed URI without treating the URI as a filename.
    source_namespace: str | None = Field(default=None, exclude=True)
    source_scope: str | None = Field(default=None, exclude=True)
    source_path: str | None = Field(default=None, exclude=True)
    name: str = ""
    description: str = ""
    etag: str | None = Field(
        default=None, description="Optional content checksum for change detection."
    )
