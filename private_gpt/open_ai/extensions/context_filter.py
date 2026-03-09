from pydantic import BaseModel, Field


class ContextFilter(BaseModel):
    docs_ids: list[str] | None = Field(
        default=None,
        examples=[["c202d5e6-7b69-4869-81cc-dd574ee8ee11"]],
    )
    collection_name: str | None = Field(
        default=None,
        description="If set, restrict retrieval to documents belonging to this collection.",
        examples=["engineering"],
    )
