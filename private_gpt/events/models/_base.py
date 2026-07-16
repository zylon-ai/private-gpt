import datetime
import logging
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic_core.core_schema import SerializerFunctionWrapHandler


def serialize_datetime(dt: datetime.datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    return dt.astimezone(datetime.UTC).isoformat().replace("+00:00", "Z")


class StandardContentProtocol:
    """Marker for Anthropic-compatible content blocks."""

    pass


class ExtendedContentProtocol(StandardContentProtocol):
    """Marker for Zylon-specific content blocks."""

    pass


class CacheControlEphemeral(BaseModel):
    type: Literal["ephemeral"] = Field()
    ttl: Literal["5m", "1h"] = Field(default="5m")
    model_config = ConfigDict(extra="allow")


class BaseContentBlock(BaseModel, StandardContentProtocol):
    model_config = ConfigDict(
        extra="allow",
        json_encoders={
            datetime.datetime: serialize_datetime,
            datetime.date: lambda v: v.isoformat() if v else None,
            datetime.time: lambda v: v.isoformat() if v else None,
        },
        json_schema_serialization_defaults_required=True,
    )

    type: str = Field(description="Content block type identifier")
    start_timestamp: datetime.datetime | None = Field(
        default=None, serialization_alias="start_timestamp"
    )
    stop_timestamp: datetime.datetime | None = Field(
        default=None, serialization_alias="stop_timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="_meta",
        serialization_alias="_meta",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_metadata(
        cls, values: dict[str, Any] | tuple[tuple[str, Any], ...] | None
    ) -> dict[str, Any] | None:
        if values is None:
            return values
        if isinstance(values, tuple):
            values = dict(values)
        if "metadata" in values:
            try:
                values["_meta"] = values.pop("metadata")
            except Exception as exc:
                logging.error(
                    "Failed to convert 'metadata' to '_meta': %s - %s", exc, values
                )
        return values

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for key in ("_meta", "metadata"):
            if not data.get(key):
                data.pop(key, None)
        for key in ("start_timestamp", "stop_timestamp"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("by_alias", True)
        return super().model_dump_json(**kwargs)

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> Self | None:
        if response_mode == "zylon":
            return self
        if response_mode == "anthropic" and not isinstance(
            self, ExtendedContentProtocol
        ):
            return self
        return None

    def __str__(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def __repr__(self) -> str:
        rest = ", ".join(
            f"{k}={v!r}"
            for k, v in self.model_dump(exclude_none=True).items()
            if k != "type"
        )
        return f"{self.__class__.__name__}(type={self.type!r}{', ' + rest if rest else ''})"

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        schema = handler(core_schema)
        if isinstance(schema, dict):
            properties = schema.get("properties")
            if isinstance(properties, dict) and "type" in properties:
                required = schema.get("required")
                if not isinstance(required, list):
                    required = []
                if "type" not in required:
                    required.append("type")
                schema["required"] = sorted(required)
        return schema


class CacheableContentBlock(BaseContentBlock, StandardContentProtocol):
    """Base for blocks that support Anthropic prompt-caching breakpoints."""

    cache_control: (
        Annotated[CacheControlEphemeral, Field(discriminator="type")] | None
    ) = Field(default=None)
