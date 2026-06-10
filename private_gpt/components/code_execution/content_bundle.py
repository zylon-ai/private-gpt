from pydantic import BaseModel, Field


class BundledFile(BaseModel):
    path: str  # relative path within the bundle e.g. "SKILL.md"
    content: bytes
    permissions: int = 0o444  # Unix permissions


class ContentBundle(BaseModel):
    canonical_path: str  # must end with "/" e.g. "/mnt/skills/foo/"
    files: list[BundledFile] = Field(default_factory=list)
    writable: bool = False
