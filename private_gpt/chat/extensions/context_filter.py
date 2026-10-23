from typing import Any

from pydantic import BaseModel, Field


class ContextFilter(BaseModel):
    """Filter by collection, artifacts and metadata in the ingested context.

    The main filter is the collection the context is part of.

    All artifacts ids are ensured to be part of the context. If any of the ids is not
    usable in the context (i.e. doesn't exist), the related task will fail.

    The metadata filter will look for all context matching the metadata, if any, and
    add it to the context. The filter is a dictionary of key-value pairs.
    The key is the metadata key, and the value is the metadata value.

    If both artifacts and metadata filters are provided, the result will be the
    intersection of the two filters.
    """

    collection: str = Field(
        description="The name of the collection to filter on.",
        examples=["collection_name"],
        default="pgpt_collection",
    )
    artifacts: list[str] | None = Field(
        default=None,
        description="Artifacts ids to filter on.",
        examples=[["artifact_id_1, artifact_id_2"]],
    )
    metadata_filter: list[dict[str, Any]] | None = Field(
        default=None,
        description="Metadata filter to apply on the context.",
        examples=[[{"key": "file_id", "operator": "==", "value": "artifact_id_1"}]],
    )
