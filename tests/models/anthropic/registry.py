from dataclasses import dataclass
from typing import Any

import anthropic.types as sdk_types
from anthropic.types.raw_message_delta_event import Delta as SDKMessageDelta

from private_gpt.events.models import (
    AudioBlock,
    BinaryBlock,
    CitationsDelta,
    ContainerUploadBlock,
    DirectCaller,
    DocumentBlock,
    ErrorBlock,
    FatalError,
    ImageBlock,
    InputJSONDelta,
    Message,
    MessageOutputDelta,
    MidConvSystemBlock,
    PingEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    ResourceBlock,
    ResourceLinkBlock,
    SearchResultBlock,
    ServerToolCaller,
    ServerToolUseBlock,
    SignatureDelta,
    SourceBlock,
    SourceDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    TLDRBlock,
    TLDRDelta,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from private_gpt.events.models._callers import ServerToolCaller20260120


@dataclass(frozen=True)
class TypeMapping:
    sdk_type: type | None  # None for Zylon-only types with no SDK equivalent
    our_type: type | None
    sdk_schema_name: str
    openapi_schema_name: str | None
    zylon_only_fields: frozenset[str]
    sdk_only_fields: frozenset[str]
    sdk_sample: dict[str, Any]
    notes: str = ""
    skip: bool = False
    # Fields present in the remote OpenAPI schema and our type but not yet in the
    # Python SDK (SDK is lagging behind the spec). Not stripped during schema
    # validation, not flagged as undeclared extensions.
    forward_compat_fields: frozenset[str] = frozenset()


_BASE_ZYLON_FIELDS: frozenset[str] = frozenset(
    {"start_timestamp", "stop_timestamp", "metadata", "cache_control"}
)

_CACHEABLE_ZYLON_FIELDS: frozenset[str] = _BASE_ZYLON_FIELDS | {"cache_control"}

# ---------------------------------------------------------------------------
# Content block registry
# ---------------------------------------------------------------------------

CONTENT_BLOCK_REGISTRY: list[TypeMapping] = [
    TypeMapping(
        sdk_type=sdk_types.TextBlock,
        our_type=TextBlock,
        sdk_schema_name="TextBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "text", "text": "Hello, world!"},
    ),
    TypeMapping(
        sdk_type=sdk_types.ThinkingBlock,
        our_type=ThinkingBlock,
        sdk_schema_name="ThinkingBlock",
        openapi_schema_name=None,
        # citations exists in our ThinkingBlock but not in SDK ThinkingBlock
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS | {"citations"},
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "thinking", "thinking": "step 1…", "signature": "sig_abc"},
    ),
    TypeMapping(
        sdk_type=sdk_types.RedactedThinkingBlock,
        our_type=RedactedThinkingBlock,
        sdk_schema_name="RedactedThinkingBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "redacted_thinking", "data": "<redacted>"},
    ),
    TypeMapping(
        sdk_type=sdk_types.ToolUseBlock,
        our_type=ToolUseBlock,
        sdk_schema_name="ToolUseBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "tool_use",
            "id": "toolu_01abc",
            "name": "get_weather",
            "input": {"location": "Paris"},
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.ServerToolUseBlock,
        our_type=ServerToolUseBlock,
        sdk_schema_name="ServerToolUseBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "server_tool_use",
            "id": "srvtoolu_01",
            "name": "web_search",
            "input": {"query": "anthropic"},
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.ContainerUploadBlock,
        our_type=ContainerUploadBlock,
        sdk_schema_name="ContainerUploadBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "container_upload", "file_id": "file_01abc"},
    ),
]


# ---------------------------------------------------------------------------
# Delta registry
# ---------------------------------------------------------------------------

DELTA_REGISTRY: list[TypeMapping] = [
    TypeMapping(
        sdk_type=sdk_types.TextDelta,
        our_type=TextDelta,
        sdk_schema_name="TextDelta",
        openapi_schema_name="TextContentBlockDelta",
        # citations on TextDelta is a Zylon extension
        zylon_only_fields=_BASE_ZYLON_FIELDS | {"citations"},
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "text_delta", "text": "incremental"},
    ),
    TypeMapping(
        sdk_type=sdk_types.InputJSONDelta,
        our_type=InputJSONDelta,
        sdk_schema_name="InputJSONDelta",
        openapi_schema_name="InputJsonContentBlockDelta",
        zylon_only_fields=_BASE_ZYLON_FIELDS | {"partial_json_obj"},
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "input_json_delta", "partial_json": '{"key":'},
    ),
    TypeMapping(
        sdk_type=sdk_types.CitationsDelta,
        our_type=CitationsDelta,
        sdk_schema_name="CitationsDelta",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "citations_delta",
            "citation": {
                "type": "char_location",
                "cited_text": "some text",
                "document_index": 0,
                "start_char_index": 0,
                "end_char_index": 9,
            },
        },
        # The current format of citations is incompatible with anthropic
        skip=True,
    ),
    TypeMapping(
        sdk_type=sdk_types.ThinkingDelta,
        our_type=ThinkingDelta,
        sdk_schema_name="ThinkingContentBlockDelta",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS | {"citations"},
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "thinking_delta", "thinking": "more reasoning…"},
    ),
    TypeMapping(
        sdk_type=sdk_types.SignatureDelta,
        our_type=SignatureDelta,
        sdk_schema_name="SignatureDelta",
        openapi_schema_name="SignatureContentBlockDelta",
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "signature_delta", "signature": "sig_xyz"},
    ),
]


# ---------------------------------------------------------------------------
# Streaming event registry
# ---------------------------------------------------------------------------

_STREAMING_ZYLON_FIELDS: frozenset[str] = _BASE_ZYLON_FIELDS | {"block_id"}

STREAMING_EVENT_REGISTRY: list[TypeMapping] = [
    TypeMapping(
        sdk_type=sdk_types.RawContentBlockStartEvent,
        our_type=RawContentBlockStartEvent,
        sdk_schema_name="RawContentBlockStartEvent",
        openapi_schema_name="ContentBlockStartEvent",
        zylon_only_fields=_STREAMING_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "content_block_start",
            "index": 0,
            "block_id": "block_001",
            "content_block": {"type": "text", "text": "x", "citations": None},
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.RawContentBlockDeltaEvent,
        our_type=RawContentBlockDeltaEvent,
        sdk_schema_name="RawContentBlockDeltaEvent",
        openapi_schema_name="ContentBlockDeltaEvent",
        zylon_only_fields=_STREAMING_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "content_block_delta",
            "index": 0,
            "block_id": "block_001",
            "delta": {"type": "text_delta", "text": "hello"},
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.RawContentBlockStopEvent,
        our_type=RawContentBlockStopEvent,
        sdk_schema_name="RawContentBlockStopEvent",
        openapi_schema_name="ContentBlockStopEvent",
        zylon_only_fields=_STREAMING_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "content_block_stop", "index": 0, "block_id": "block_001"},
    ),
    TypeMapping(
        sdk_type=sdk_types.RawMessageStartEvent,
        our_type=RawMessageStartEvent,
        sdk_schema_name="RawMessageStartEvent",
        openapi_schema_name="MessageStartEvent",
        zylon_only_fields=_STREAMING_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "message_start",
            "message": {
                "id": "msg_01abc",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-6",
                "stop_details": None,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "container": None,
                "usage": {
                    "cache_creation": None,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": None,
                    "inference_geo": None,
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "output_tokens_details": None,
                    "server_tool_use": None,
                    "service_tier": None,
                },
            },
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.RawMessageDeltaEvent,
        our_type=RawMessageDeltaEvent,
        sdk_schema_name="RawMessageDeltaEvent",
        openapi_schema_name="MessageDeltaEvent",
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "message_delta",
            "delta": {
                "container": None,
                "stop_details": None,
                "stop_reason": "end_turn",
                "stop_sequence": None,
            },
            "usage": {
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "input_tokens": None,
                "output_tokens": 42,
                "output_tokens_details": None,
                "server_tool_use": None,
            },
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.RawMessageStopEvent,
        our_type=RawMessageStopEvent,
        sdk_schema_name="RawMessageStopEvent",
        openapi_schema_name="MessageStopEvent",
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "message_stop"},
    ),
]


# ---------------------------------------------------------------------------
# Message / Usage registry
# ---------------------------------------------------------------------------

_USAGE_SDK_ONLY: frozenset[str] = frozenset(
    {
        "cache_creation",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "inference_geo",
        "server_tool_use",
        "service_tier",
    }
)

MESSAGE_REGISTRY: list[TypeMapping] = [
    TypeMapping(
        sdk_type=sdk_types.Message,
        our_type=Message,
        sdk_schema_name="Message",
        openapi_schema_name=None,
        zylon_only_fields=frozenset({"cache_control"}),
        sdk_only_fields=frozenset(),
        sdk_sample={
            "id": "msg_01abc",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi", "citations": None}],
            "model": "claude-sonnet-4-6",
            "stop_details": None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "container": None,
            "usage": {
                "cache_creation": None,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "inference_geo": None,
                "input_tokens": 10,
                "output_tokens": 5,
                "output_tokens_details": None,
                "server_tool_use": None,
                "service_tier": None,
            },
        },
    ),
    TypeMapping(
        sdk_type=sdk_types.Usage,
        our_type=Usage,
        sdk_schema_name="Usage",
        openapi_schema_name=None,
        zylon_only_fields=frozenset(),
        sdk_only_fields=_USAGE_SDK_ONLY,
        forward_compat_fields=frozenset({"output_tokens_details"}),
        sdk_sample={
            "cache_creation": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "inference_geo": None,
            "input_tokens": 42,
            "output_tokens": 17,
            "output_tokens_details": None,
            "server_tool_use": None,
            "service_tier": None,
        },
    ),
]

ZYLON_ONLY_REGISTRY: list[TypeMapping] = [
    # -- Content blocks -----------------------------------------------------
    TypeMapping(
        sdk_type=sdk_types.DocumentBlock,
        our_type=DocumentBlock,
        sdk_schema_name="DocumentBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS | {"context"},
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "…"},
        },
    ),
    TypeMapping(
        sdk_type=None,
        our_type=ImageBlock,
        sdk_schema_name="ImageBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "aGVsbG8=",
            },
        },
    ),
    TypeMapping(
        sdk_type=None,
        our_type=AudioBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-internal base64 audio block; no Anthropic SDK equivalent",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=SearchResultBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS | {"citations"},
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only search result content block",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=MidConvSystemBlock,
        sdk_schema_name="RequestMidConvSystemBlock",
        openapi_schema_name="RequestMidConvSystemBlock",
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "mid_conv_system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        notes="In the Anthropic OpenAPI spec (InputContentBlock) but not yet in the Python SDK",
    ),
    TypeMapping(
        sdk_type=None,
        our_type=BinaryBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only arbitrary binary payload block (base64-encoded)",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=ResourceLinkBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only external resource reference by URI (not embedded)",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=ResourceBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only embedded resource block with metadata",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=SourceBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only RAG context attribution block (surfaced document chunks)",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=TLDRBlock,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only condensed summary block",
        skip=True,
    ),
    # -- Tool results -------------------------------------------------------
    TypeMapping(
        sdk_type=None,  # SDK only has ToolResultBlockParam (TypedDict) — not a BaseModel
        our_type=ToolResultBlock,
        sdk_schema_name="ToolResultBlock",
        openapi_schema_name=None,
        zylon_only_fields=_CACHEABLE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "tool_result",
            "tool_use_id": "toolu_01abc",
            "content": "ok",
            "is_error": False,
        },
    ),
    # -- Deltas -------------------------------------------------------------
    TypeMapping(
        sdk_type=None,
        our_type=SourceDelta,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only incremental RAG source attribution delta",
        skip=True,
    ),
    TypeMapping(
        sdk_type=None,
        our_type=TLDRDelta,
        sdk_schema_name="",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS,
        sdk_only_fields=frozenset(),
        sdk_sample={},
        notes="Zylon-only incremental TLDR summary delta",
        skip=True,
    ),
    # -- Streaming events ---------------------------------------------------
    TypeMapping(
        sdk_type=None,
        our_type=PingEvent,
        sdk_schema_name="PingEvent",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS | {"type"},
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "ping"},
    ),
    # -- Message ------------------------------------------------------------
    TypeMapping(
        sdk_type=SDKMessageDelta,
        our_type=MessageOutputDelta,
        sdk_schema_name="MessageDelta",
        openapi_schema_name=None,
        zylon_only_fields=_BASE_ZYLON_FIELDS
        | {"id", "type", "role", "content", "model", "usage"},
        sdk_only_fields=frozenset(),
        sdk_sample={
            "container": None,
            "stop_details": None,
            "stop_reason": "end_turn",
            "stop_sequence": None,
        },
    ),
    # -- Errors -------------------------------------------------------------
    TypeMapping(
        sdk_type=sdk_types.APIErrorObject,
        our_type=ErrorBlock,
        sdk_schema_name="APIErrorObject",
        openapi_schema_name="APIError",
        zylon_only_fields=_BASE_ZYLON_FIELDS | {"detail"},
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "api_error", "message": "Internal server error"},
    ),
    TypeMapping(
        sdk_type=sdk_types.ErrorResponse,
        our_type=FatalError,
        sdk_schema_name="ErrorResponse",
        openapi_schema_name=None,
        zylon_only_fields=frozenset({"exception"}),
        sdk_only_fields=frozenset(),
        sdk_sample={
            "type": "error",
            "request_id": None,
            "error": {"type": "api_error", "message": "Internal server error"},
        },
    ),
    # -- Callers ------------------------------------------------------------
    TypeMapping(
        sdk_type=sdk_types.DirectCaller,
        our_type=DirectCaller,
        sdk_schema_name="DirectCaller",
        openapi_schema_name=None,
        zylon_only_fields=frozenset(),
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "direct"},
    ),
    TypeMapping(
        sdk_type=sdk_types.ServerToolCaller,
        our_type=ServerToolCaller,
        sdk_schema_name="ServerToolCaller",
        openapi_schema_name=None,
        zylon_only_fields=frozenset(),
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "code_execution_20250825", "tool_id": "srvtoolu_01"},
    ),
    TypeMapping(
        sdk_type=sdk_types.ServerToolCaller20260120,
        our_type=ServerToolCaller20260120,
        sdk_schema_name="ServerToolCaller20260120",
        openapi_schema_name="ServerToolCaller_20260120",
        zylon_only_fields=frozenset(),
        sdk_only_fields=frozenset(),
        sdk_sample={"type": "code_execution_20260120", "tool_id": "srvtoolu_02"},
    ),
]


# ---------------------------------------------------------------------------
# Combined view
# ---------------------------------------------------------------------------

ALL_REGISTRIES: dict[str, list[TypeMapping]] = {
    "content_blocks": CONTENT_BLOCK_REGISTRY,
    "deltas": DELTA_REGISTRY,
    "streaming_events": STREAMING_EVENT_REGISTRY,
    "message": MESSAGE_REGISTRY,
    "zylon_only": ZYLON_ONLY_REGISTRY,
}

ALL_MAPPINGS: list[TypeMapping] = [
    m for registry in ALL_REGISTRIES.values() for m in registry if not m.skip
]
