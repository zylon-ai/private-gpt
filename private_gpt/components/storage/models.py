from pydantic import BaseModel, Field


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
