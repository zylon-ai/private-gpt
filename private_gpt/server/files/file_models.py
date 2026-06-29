from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FileScope(BaseModel):
    """Scope that a file belongs to (always a session)."""

    id: str = Field(
        description="Session / container identifier that owns this file.",
        examples=["session-abc123"],
    )
    type: Literal["session"] = Field(
        default="session",
        description="Object type discriminator, always 'session'.",
        examples=["session"],
    )


class FileMetadata(BaseModel):
    """Metadata for a file stored in a session (upload or sandbox output)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "uploads/data.csv",
                    "created_at": "2024-01-15T10:30:00Z",
                    "filename": "data.csv",
                    "mime_type": "text/csv",
                    "size_bytes": 4096,
                    "type": "file",
                    "downloadable": False,
                    "scope": {"id": "session-abc123", "type": "session"},
                }
            ]
        }
    )

    id: str = Field(
        description=(
            "Relative path of the file within the session, e.g. 'uploads/data.csv' or "
            "'outputs/result.png'. Use this value as `file_id` in subsequent requests."
        ),
        examples=["uploads/data.csv"],
    )
    created_at: datetime = Field(
        description="ISO-8601 timestamp when the file was created or last modified.",
        examples=["2024-01-15T10:30:00Z"],
    )
    filename: str = Field(
        description="Filename derived from the path.",
        examples=["data.csv"],
    )
    mime_type: str = Field(
        description="MIME type detected from the file content.",
        examples=["text/csv"],
    )
    size_bytes: int = Field(
        description="File size in bytes.",
        examples=[4096],
    )
    type: Literal["file"] = Field(
        default="file",
        description="Object type discriminator, always 'file'.",
        examples=["file"],
    )
    downloadable: bool = Field(
        description="True for sandbox output files; False for uploaded input files.",
        examples=[False],
    )
    scope: FileScope = Field(
        description="Session scope this file belongs to.",
    )


class DeletedFile(BaseModel):
    """Confirmation that a file was deleted."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "uploads/data.csv",
                    "type": "file_deleted",
                }
            ]
        }
    )

    id: str = Field(
        description="Relative path of the file that was deleted.",
        examples=["uploads/data.csv"],
    )
    type: Literal["file_deleted"] = Field(
        default="file_deleted",
        description="Object type discriminator, always 'file_deleted'.",
        examples=["file_deleted"],
    )


class FileListResponse(BaseModel):
    """Paginated list of files in a session."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": [
                        {
                            "id": "uploads/data.csv",
                            "created_at": "2024-01-15T10:30:00Z",
                            "filename": "data.csv",
                            "mime_type": "text/csv",
                            "size_bytes": 4096,
                            "type": "file",
                            "downloadable": False,
                            "scope": {"id": "session-abc123", "type": "session"},
                        }
                    ],
                    "first_id": "uploads/data.csv",
                    "last_id": "outputs/result.png",
                    "has_more": False,
                }
            ]
        }
    )

    data: list[FileMetadata] = Field(
        default_factory=list,
        description="List of file metadata objects for the current page.",
    )
    first_id: str | None = Field(
        default=None,
        description="ID of the first file in the current page, used for cursor-based pagination.",
        examples=["uploads/data.csv"],
    )
    last_id: str | None = Field(
        default=None,
        description="ID of the last file in the current page, used for cursor-based pagination.",
        examples=["outputs/result.png"],
    )
    has_more: bool = Field(
        default=False,
        description="True when there are additional pages of results beyond this one.",
        examples=[False],
    )
