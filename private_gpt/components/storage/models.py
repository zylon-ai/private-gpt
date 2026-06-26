from datetime import datetime

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Metadata for a single file returned by stat_file / list_files_meta."""

    path: str = Field(
        description="Relative path within the prefix, e.g. 'uploads/data.csv'."
    )
    size_bytes: int = Field(description="File size in bytes.")
    created_at: datetime = Field(description="Last-modified timestamp (UTC).")
    mime_type: str = Field(description="MIME type of the file content.")


class StoredFile(BaseModel):
    path: str = Field(
        description="Relative file path inside the storage prefix.",
        min_length=1,
        max_length=512,
    )
    content: bytes = Field(description="Raw file content bytes.")
    mime_type: str | None = Field(
        default=None,
        description="Optional MIME type stored with the object.",
    )
