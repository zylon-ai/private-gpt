from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from private_gpt.components.llm.custom.base import (
    StructuredOutputsParams,
    normalize_structured_outputs,
)
from private_gpt.components.llm.models import ReasoningEffort


class ChatLLMParameters(BaseModel):
    """Typed parameters passed to the chat LLM.

    These values are kept in the resumable chat state, so they must remain
    typed when the state is dumped to JSON and restored for a later iteration.
    Optional fields preserve the previous dict behavior: values not supplied by
    the request are omitted when invoking the LLM.
    """

    seed: int | None = Field(default=None)
    min_p: float | None = Field(default=None)
    top_p: float | None = Field(default=None)
    temperature: float | None = Field(default=None)
    top_k: int | None = Field(default=None)
    repetition_penalty: float | None = Field(default=None)
    presence_penalty: float | None = Field(default=None)
    frequency_penalty: float | None = Field(default=None)
    max_tokens: int | None = Field(default=None)
    skip_special_tokens: bool | None = Field(default=None)
    reasoning_effort: ReasoningEffort | None = Field(default=None)
    structured_outputs: StructuredOutputsParams | None = Field(default=None)

    @field_validator("structured_outputs", mode="before")
    @classmethod
    def _normalize_structured_outputs(
        cls, value: Any
    ) -> StructuredOutputsParams | None:
        return normalize_structured_outputs(value)

    # Keep accepting provider-specific scalar parameters that are not known
    # here yet. Known parameters above are explicitly typed so nested values
    # such as enums and BaseModels are restored correctly.
    model_config = ConfigDict(extra="allow")

    def as_kwargs(self) -> dict[str, Any]:
        """Return invocation kwargs without recursively serializing values."""
        values = {
            **(self.__pydantic_extra__ or {}),
            **self.__dict__,
        }
        return {key: value for key, value in values.items() if value is not None}

    @field_serializer("structured_outputs", when_used="json")
    def _serialize_structured_outputs(
        self, value: StructuredOutputsParams | None
    ) -> dict[str, Any] | None:
        """Serialize the state shape, not StructuredOutputsParams' API shape.

        StructuredOutputsParams intentionally serializes ``json_schema`` as
        ``json`` when calling an LLM. For checkpoint data we need the model
        field name so Pydantic can validate it back into the nested model.
        """
        return value.__dict__.copy() if value is not None else None
