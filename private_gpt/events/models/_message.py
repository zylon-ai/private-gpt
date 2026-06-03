import enum
from datetime import datetime
from typing import Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from private_gpt.events.models._base import StandardContentProtocol
from private_gpt.events.models._tool_result_blocks import ContentBlockType


class OutputTokensDetails(BaseModel):
    """Breakdown of output tokens by category."""

    reasoning_tokens: int | None = Field(
        default=None, description="Output tokens spent on internal reasoning."
    )


class CacheCreation(BaseModel):
    """Token counts cached by ephemeral policy."""

    ephemeral_1h_input_tokens: int = Field(
        description="Input tokens cached with 1-hour TTL."
    )
    ephemeral_5m_input_tokens: int = Field(
        description="Input tokens cached with 5-minute TTL."
    )


class ServerToolUsage(BaseModel):
    """Usage counters for server-side tools."""

    web_fetch_requests: int = Field(description="Number of web fetch requests used.")
    web_search_requests: int = Field(description="Number of web search requests used.")


class Container(BaseModel):
    """Container handle for request reuse."""

    id: str = Field(description="Container identifier.")
    expires_at: datetime = Field(description="Container expiration timestamp.")


class RefusalStopDetails(BaseModel):
    """Additional metadata when generation stops due to refusal."""

    type: Literal["refusal"] = Field(description="Stop-details discriminator.")
    category: Literal["cyber", "bio"] | None = Field(
        default=None, description="Optional refusal category."
    )
    explanation: str | None = Field(
        default=None, description="Optional human-readable refusal explanation."
    )


class Usage(BaseModel):
    """Token usage statistics."""

    cache_creation: CacheCreation | None = Field(
        default=None, description="Cache creation details, when present."
    )
    cache_creation_input_tokens: int | None = Field(
        default=None, description="Number of input tokens written to cache."
    )
    cache_read_input_tokens: int | None = Field(
        default=None, description="Number of input tokens read from cache."
    )
    inference_geo: str | None = Field(
        default=None, description="Inference region code, when available."
    )
    input_tokens: int | None = Field(
        default=None, description="Input token count for this response."
    )
    output_tokens: int | None = Field(
        default=None, description="Output token count for this response."
    )
    output_tokens_details: OutputTokensDetails | None = Field(
        default=None, description="Breakdown of output tokens by category."
    )
    server_tool_use: ServerToolUsage | None = Field(
        default=None, description="Usage counters for server-side tool calls."
    )
    service_tier: Literal["standard", "priority", "batch"] | None = Field(
        default=None, description="Service tier used for this response."
    )

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


class StopReasonEnum(enum.StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"

    @classmethod
    def convert_from_vllm(
        cls, vllm_reason: str | None
    ) -> Union["StopReasonEnum", None]:
        match vllm_reason:
            case "stop":
                return cls.END_TURN
            case "length":
                return cls.MAX_TOKENS
        return None

    @classmethod
    def convert_from_openai(
        cls, openai_reason: str | None
    ) -> Union["StopReasonEnum", None]:
        match openai_reason:
            case "stop":
                return cls.END_TURN
            case "length":
                return cls.MAX_TOKENS
            case "tool_calls":
                return cls.TOOL_USE
            case "content_filter":
                return cls.REFUSAL
        return None


class Message(BaseModel, StandardContentProtocol):
    """Anthropic-compatible message response."""

    id: str = Field(
        default_factory=lambda: f"msg_{uuid4().hex}",
        description="Message identifier.",
    )
    type: Literal["message"] = Field(default="message", description="Object type.")
    role: Literal["assistant"] = Field(
        default="assistant", description="Message author role."
    )
    content: list[ContentBlockType] = Field(
        default_factory=list, description="Assistant content blocks."
    )
    model: str = Field(default="private-gpt", description="Model name used.")
    container: Container | None = Field(
        default=None, description="Optional execution container."
    )
    stop_details: RefusalStopDetails | None = Field(
        default=None, description="Optional structured stop details."
    )
    stop_reason: str | None = Field(default=None, description="Message stop reason.")
    stop_sequence: str | None = Field(
        default=None, description="Matched stop sequence, if any."
    )
    usage: Usage = Field(default_factory=Usage, description="Token usage stats.")


class MessageOutputDelta(BaseModel, StandardContentProtocol):
    """Partial message update emitted during streaming."""

    id: str | None = Field(default=None, description="Message identifier delta.")
    type: str | None = Field(default=None, description="Object type delta.")
    role: str | None = Field(default=None, description="Role delta.")
    content: list[ContentBlockType] | None = Field(
        default=None, description="Content delta."
    )
    model: str | None = Field(default=None, description="Model delta.")
    container: Container | None = Field(default=None, description="Container delta.")
    stop_details: RefusalStopDetails | None = Field(
        default=None, description="Stop-details delta."
    )
    stop_reason: str | None = Field(default=None, description="Stop-reason delta.")
    stop_sequence: str | None = Field(default=None, description="Stop-sequence delta.")
    usage: Usage | None = Field(default=None, description="Usage delta.")
