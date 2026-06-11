from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field


class BundledFile(BaseModel):
    path: str  # relative path within the bundle e.g. "SKILL.md"
    content: bytes
    permissions: int = 0o444  # Unix permissions


class ContentBundle(BaseModel):
    canonical_path: str  # must end with "/" e.g. "/mnt/skills/foo/"
    files: list[BundledFile] = Field(default_factory=list)
    writable: bool = False


class StoredBundle(ContentBundle):
    """Content that lives in object storage — by reference, not by value.

    Mounters bind ``storage_prefix`` directly from the storage host path when
    one is available (read-only, no duplication); ``fetch`` is the copy
    fallback, called only when the content must be materialised by hand.
    """

    storage_prefix: str  # e.g. "skills/{collection}/{skill_id}/{version_id}"
    fetch: Callable[[], Awaitable[list[BundledFile]]] = Field(exclude=True)
