import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, NamedTuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.server.ingest.uri_loader import load_file_from_uri


class BinaryContent(NamedTuple):
    """Binary content with filename."""

    data: BinaryIO
    filename: str


def _is_valid_base64(data: str) -> bool:
    """Validate if string is properly base64 encoded."""
    try:
        cleaned = data.strip()
        if not cleaned:
            return False
        decoded = base64.b64decode(cleaned, validate=True)
        return base64.b64encode(decoded).decode("ascii") == cleaned
    except Exception:
        return False


def _is_valid_uri(uri: str) -> bool:
    """Validate if string is a valid URI."""
    try:
        result = urlparse(uri.strip())
        return all([result.scheme, result.netloc])
    except Exception:
        return False


class Artifact(BaseModel, ABC):
    """Abstract base class for all input types."""

    @abstractmethod
    def extract_filename(self, fallback_name: str | None = None) -> str:
        """Extract filename from the input."""
        pass

    @abstractmethod
    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        """Convert to BinaryContent with filename."""
        pass

    def to_binary(self) -> BinaryIO:
        """Convert to BinaryIO (legacy method)."""
        return self.to_binary_content().data


class FileArtifact(Artifact):
    """Input for base64 encoded files."""

    type: Literal["file"] = Field(
        default="file", description="Input type discriminator"
    )
    value: str = Field(..., description="Base64 encoded file content")

    @field_validator("value")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        if not _is_valid_base64(v):
            raise ValueError("File input requires valid base64 encoded content")
        return v

    def extract_filename(self, fallback_name: str | None = None) -> str:
        return fallback_name or "uploaded_file"

    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        decoded = base64.b64decode(self.value)
        extracted_filename = self.extract_filename(filename)
        return BinaryContent(io.BytesIO(decoded), extracted_filename)


class UriArtifact(Artifact):
    """Input for remote URIs."""

    type: Literal["uri"] = Field(default="uri", description="Input type discriminator")
    value: str = Field(..., description="URI to download from")

    def extract_filename(self, fallback_name: str | None = None) -> str:
        if fallback_name:
            return fallback_name
        try:
            # Attempt to extract filename from the URL
            parsed = urlparse(self.value)
            filename = Path(parsed.path).name
            return filename if filename else fallback_name or "downloaded_file"
        except Exception:
            return fallback_name or "downloaded_file"

    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        binary_data = load_file_from_uri(self.value)
        extracted_filename = self.extract_filename(filename)
        return BinaryContent(binary_data, extracted_filename)

    def is_base64(self) -> bool:
        """Check if the URI is a base64 encoded string."""
        return _is_valid_base64(self.value)

    def is_s3(self) -> bool:
        """Check if the URI is an S3 URI."""
        return self.value.strip().startswith("s3://")

    def is_from_s3_bucket(self, bucket_name: str) -> bool:
        """Check if the URI is from the specified S3 bucket."""
        if not self.is_s3():
            return False
        s3_path = self.value.strip()[5:]
        s3_components = s3_path.split("/", 1)
        s3_bucket = s3_components[0]
        return s3_bucket == bucket_name


class TextArtifact(Artifact):
    """Input for plain text content."""

    type: Literal["text"] = Field(
        default="text", description="Input type discriminator"
    )
    value: str = Field(..., description="Plain text content")

    def extract_filename(self, fallback_name: str | None = None) -> str:
        return fallback_name or "text_content.txt"

    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        encoded = self.value.encode("utf-8")
        extracted_filename = self.extract_filename(filename)
        return BinaryContent(io.BytesIO(encoded), extracted_filename)


class IngestedArtifact(Artifact):
    """Input for already ingested content."""

    type: Literal["ingested_artifact"] = Field(
        default="ingested_artifact", description="Input type discriminator"
    )
    context_filter: ContextFilter = Field(
        ..., description="Already processed ContextFilter"
    )

    def extract_filename(self, fallback_name: str | None = None) -> str:
        return fallback_name or "ingested_content"

    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        raise ValueError("Ingested data cannot be converted to binary directly")

    def get_context_filter(self) -> ContextFilter:
        """Get the ContextFilter directly."""
        return self.context_filter


class SqlDatabaseArtifact(Artifact):
    """Input for SQL database content."""

    type: Literal["sql_database"] = Field(
        default="sql_database", description="Input type discriminator"
    )
    connection_string: str = Field(..., description="SQL database connection string")

    schemas: list[str] | None = Field(
        default=None,
        description="List of specific schemas to include, if None includes all non-system schemas",
    )

    ssl: bool = Field(
        default=False,
        description="Whether to use SSL for the database connection",
    )

    enable_tables: bool = Field(
        default=True,
        description="Whether to include tables in the inspection",
    )

    enable_views: bool = Field(
        default=True,
        description="Whether to include views in the inspection",
    )

    enable_functions: bool = Field(
        default=True,
        description="Whether to include functions in the inspection",
    )

    enable_procedures: bool = Field(
        default=True,
        description="Whether to include stored procedures in the inspection",
    )

    description: str = Field(
        default="",
        description="Optional description of the database",
    )

    def extract_filename(self, fallback_name: str | None = None) -> str:
        return fallback_name or "sql_database_content"

    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        raise ValueError("SQL database content cannot be converted to binary directly")


class SkillArtifact(Artifact):
    """Input for skill activation using SkillFilter."""

    type: Literal["skill"] = Field(
        default="skill", description="Input type discriminator"
    )
    skill_filter: SkillFilter = Field(
        ...,
        description=(
            "Skill filter declaring the collection and optional "
            "skill_or_version_ids whitelist."
        ),
    )

    def extract_filename(self, fallback_name: str | None = None) -> str:
        return fallback_name or "skill_context"

    def to_binary_content(self, filename: str | None = None) -> BinaryContent:
        raise ValueError("Skill artifact cannot be converted to binary directly")


IngestableArtifactType = Annotated[
    FileArtifact | UriArtifact | TextArtifact,
    Field(discriminator="type"),
]
ArtifactType = Annotated[
    FileArtifact
    | UriArtifact
    | TextArtifact
    | IngestedArtifact
    | SqlDatabaseArtifact
    | SkillArtifact,
    Field(discriminator="type"),
]
