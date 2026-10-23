import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.multi_modal_llms import MultiModalLLMMetadata
from llama_index.core.types import Model
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from private_gpt.components.llm.prompt_styles.prompt_style_base import (
    CompletionToPromptProtocol,
    MessageToPromptProtocol,
)


class StructuredOutputsParams(BaseModel):
    """One of these fields will be used to build a logit processor."""

    json_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for the output. Can be a string or a dictionary.",
    )
    regex: str | None = Field(
        default=None, description="Regular expression to match the output. "
    )
    choice: list[str] | None = Field(
        default=None,
        description="List of choices for the output. "
        "If provided, the model will choose one of these choices.",
    )
    grammar: str | None = Field(
        default=None,
        description="Grammar for the output. "
        "If provided, the model will generate output based on this grammar.",
    )
    json_object: bool | None = Field(
        default=None,
        description="Whether to enforce the output to be a JSON object. "
        "Only applicable if 'json' is provided.",
    )

    # Other options
    disable_fallback: bool = Field(
        default=False, description="Whether to disable fallback decoding. "
    )
    disable_any_whitespace: bool = Field(
        default=False, description="Whether to disable any whitespace in the output. "
    )
    disable_additional_properties: str | None = Field(
        default=None,
        description="If set to 'remove', additional properties in JSON output will be removed. ",
    )
    structural_tag: str | None = Field(
        default=None, description="Structural tag to guide decoding. "
    )

    @staticmethod
    def from_optional(
        output_cls: dict[str, Any] | Model | type[Model] | str | None = None,
        json_schema: dict[str, Any] | None = None,
        regex: str | None = None,
        choice: list[str] | None = None,
        grammar: str | None = None,
    ) -> Optional["StructuredOutputsParams"]:
        if all(arg is None for arg in (output_cls, regex, choice, grammar)):
            return None

        if isinstance(output_cls, TypeAdapter):
            json_schema = output_cls.json_schema()
        elif isinstance(output_cls, BaseModel | type(BaseModel)):
            json_schema = output_cls.model_json_schema()

        return StructuredOutputsParams(
            json_schema=json_schema,
            regex=regex,
            choice=choice,
            grammar=grammar,
        )

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        """Custom serializer to handle metadata."""
        data: dict[str, Any] = handler(self)
        if data.get("json_schema"):
            json_schema = data.pop("json_schema")
            data["json"] = json_schema
        return data

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        exclude_none = kwargs.pop("exclude_none", True)
        return super().model_dump(exclude_none=exclude_none, **kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        exclude_none = kwargs.pop("exclude_none", True)
        return super().model_dump_json(exclude_none=exclude_none, **kwargs)


def normalize_structured_outputs(
    structured_outputs: (StructuredOutputsParams | Mapping[str, Any] | str | None),
) -> StructuredOutputsParams | None:
    """Normalize structured-output values crossing untyped boundaries.

    Chat parameters can be restored from serialized checkpoint data, and some
    API serializers use ``json`` instead of the model's internal
    ``json_schema`` field. Normalize those representations before backend
    specific code accesses the typed fields.
    """
    if structured_outputs is None:
        return None
    if isinstance(structured_outputs, StructuredOutputsParams):
        return structured_outputs

    if isinstance(structured_outputs, str):
        try:
            structured_outputs = json.loads(structured_outputs)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "structured_outputs must be a StructuredOutputsParams, "
                "mapping, JSON object string, or None"
            ) from exc
        if not isinstance(structured_outputs, Mapping):
            raise TypeError(
                "structured_outputs JSON must decode to an object; "
                f"got {type(structured_outputs).__name__}"
            )

    if isinstance(structured_outputs, Mapping):
        values = dict(structured_outputs)
        if "json" in values and "json_schema" not in values:
            values["json_schema"] = values.pop("json")
        return StructuredOutputsParams.model_validate(values)

    raise TypeError(
        "structured_outputs must be a StructuredOutputsParams, mapping, "
        "JSON object string, or None; "
        f"got {type(structured_outputs).__name__}"
    )


class SamplingParameters(BaseModel):
    """Sampling parameters for the model."""

    seed: int = Field(default=0, description="Random seed for reproducibility.")
    min_p: float = Field(
        default=0.0, description="Minimum probability for nucleus sampling."
    )
    top_p: float = Field(default=1.0, description="Top-p value for nucleus sampling.")
    temperature: float = Field(default=1.0, description="Temperature for sampling.")
    top_k: int = Field(default=50, description="Top-k value for sampling.")
    repetition_penalty: float = Field(
        default=1.0, description="Repetition penalty for sampling."
    )
    presence_penalty: float = Field(
        default=0.0, description="Presence penalty for sampling."
    )
    frequency_penalty: float = Field(
        default=0.0, description="Frequency penalty for sampling."
    )
    max_tokens: int = Field(
        default=512, description="Maximum number of tokens to generate."
    )
    skip_special_tokens: bool = Field(
        default=False,
        description="Whether to skip special tokens in the output.",
    )
    structured_outputs: StructuredOutputsParams | None = Field(
        default=None,
        description="Parameters for guided decoding, such as JSON schema, regex, choices, or grammar.",
    )

    @staticmethod
    def valid_keys() -> list[str]:
        """Return the valid keys for sampling parameters."""
        return [
            "seed",
            "min_p",
            "top_p",
            "temperature",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "skip_special_tokens",
            "structured_outputs",
        ]


def _default_factory() -> Any:
    raise NotImplementedError(
        "No default factory provided for this field. Please provide a default factory or set a default value."
    )


class ZylonLLM(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_to_input: MessageToPromptProtocol = Field(
        default_factory=_default_factory,
        description="Message to input prompt.",
    )
    completion_to_input: CompletionToPromptProtocol = Field(
        default_factory=_default_factory,
        description="Completion to input prompt.",
    )

    @abstractmethod
    def get_metadata(self, **kwargs: Any) -> LLMMetadata | MultiModalLLMMetadata:
        pass
