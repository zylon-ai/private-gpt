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
    """Where a mount's content comes from during development hydration.

    The host path is the source of truth once a sandbox exists; ``fetch()``
    only (re)hydrates the host file/folder from ``uri`` in local development.
    The URI may be anything ``load_file_from_uri`` understands — ``s3://``,
    ``https://``, ``data:`` or a local disk path.
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
    fetch_ref: str | None = Field(
        default=None,
        description=(
            "Dotted path to a factory ``(uri, filename) -> fetch callable``. "
            "Used to rebuild a storage-aware fetch after JSON round-trips."
        ),
    )

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
            fetch_ref = data.get("fetch_ref")
            fetch = (
                _fetch_from_ref(fetch_ref, sources)
                if isinstance(fetch_ref, str) and fetch_ref
                else None
            )
            return {
                **data,
                "fetch": fetch or _fetch_from_uris(sources),
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


def _fetch_from_ref(
    fetch_ref: str,
    sources: list[tuple[str, str | None]],
) -> Callable[[], Awaitable[list[MountFile]]] | None:
    """Build a fetch from a dotted-path factory, when one is configured."""
    if not fetch_ref:
        return None
    module_path, _, attr = fetch_ref.rpartition(":")
    if not module_path or not attr:
        return None
    import importlib

    try:
        factory = getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError):
        return None
    uri, filename = sources[0]
    return factory(uri, filename)


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


class MountSource(BaseModel):
    """Storage identity of a mount's content.

    ``(namespace, scope, path)`` locates the content inside a filesystem
    namespace (e.g. ``artifacts`` / thread-id / ``_content.md``). It is
    deliberately storage-neutral and survives JSON round-trips, so hydration
    and mount-change detection never treat a signed URI as identity.
    """

    namespace: str | None = None
    scope: str | None = None
    path: str | None = None


class Mount(BaseModel):
    """A single bind volume: one host path exposed at one exact container path.

    Mirrors a Docker ``-v`` mapping. There are exactly two kinds of mounts:

    - **Folder mount**: ``target`` ends with ``/`` (e.g. ``/mnt/artifacts/``).
      The whole host folder is visible; a more specific mount may shadow a
      subtree of it.
    - **File mount**: ``target`` has a filename (e.g.
      ``/home/agent/workspace/potato.md``). The single host file is bound at
      exactly that path, even when the path lives inside another mount.

    - ``host_path``: the host file or folder backing this mount. Always set by
      the time a sandbox is created; ``uri_source`` only exists to (re)fill it
      during development hydration.
    - ``uri_source``: hydration origin (``s3://``, ``https://``, ``data:`` or
      a local disk path). Never used to write into a running sandbox.
    - ``source``: storage identity of the content (``MountSource``). Kept in
      the model so tool configs round-trip through JSON without losing it.
    - ``etag``: optional content checksum from the Backend, used as the
      hydration change-detection key.
    """

    target: str  # e.g. "/home/agent/" (folder) or "/home/agent/a.md" (file)
    access: AccessMode = "ro"
    host_path: Path | None = None  # exact host file/folder backing the bind
    uri_source: UriSource | None = None  # hydration origin (dev only)
    source: MountSource | None = None
    name: str = ""
    description: str = ""
    etag: str | None = Field(
        default=None, description="Optional content checksum for change detection."
    )

    @property
    def is_folder(self) -> bool:
        """True when this mount exposes a folder (target ends with '/')."""
        return self.target.endswith("/")
