import enum
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import groupby
from typing import Annotated, Any, Literal

from annotated_types import Ge, Le
from llama_index.core.base.llms.types import (
    AudioBlock as LIAudioBlock,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.types import (
    ContentBlock as LIContentBlock,
)
from llama_index.core.base.llms.types import (
    ImageBlock as LIImageBlock,
)
from llama_index.core.base.llms.types import (
    TextBlock as LITextBlock,
)
from llama_index.core.llms.llm import ToolSelection
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from private_gpt.chat.extensions.citation import ZylonCitation
from private_gpt.components.tools.tool_names import resolve_internal_tool_name
from private_gpt.events.models import (
    AudioBlock,
    BaseContentBlock,
    CacheControlEphemeral,
    ContentBlockType,
    ImageBlock,
    MidConvSystemBlock,
    TextBlock,
    TLDRBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from private_gpt.server.ingest.uri_loader import load_file_from_uri
from private_gpt.server.utils.artifact_input import ArtifactType
from private_gpt.settings.settings import settings


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


class Citations(BaseModel):
    """Configuration for citation generation in AI responses."""

    enabled: bool = Field(default=False, description="Enable citations in responses")
    known_citations: list[ZylonCitation] | None = Field(
        default=None,
        description="List of known citations to use in the response",
    )


class SystemExtensions(enum.StrEnum):
    """Enumeration of supported system extensions."""

    ZYLON = "zylon"


class BlobVisibilityMode(enum.StrEnum):
    """Controls visibility and storage mode for binary large objects (blobs)."""

    BINARY = "binary"  # Returns raw base64 data
    INTERNAL = "internal"  # Uploads to private S3, returns internal URI
    PUBLIC = "public"  # Uploads to public S3, returns public URL


class PromptConfig(BaseModel):
    """Controls which platform-level prompt features are injected.

    These flags represent optional AI features adding internal instructions
    to the system prompt. All flags default to ``False`` — opt-in explicitly.
    """

    tools: bool = Field(
        default=False,
        description="Enable per-tool instruction injection for all available tools.",
    )
    citations: bool = Field(
        default=False,
        description="Enable citation formatting guidelines injection.",
    )
    thinking: bool = Field(
        default=False,
        description="Enable thinking/reasoning guidelines when thinking is enabled.",
    )
    code_execution: bool = Field(
        default=False,
        description=(
            "Enable code execution environment instructions (filesystem layout, "
            "available paths) when any code execution tool is present."
        ),
    )
    skills: bool = Field(
        default=False,
        description=(
            "Enable skill management instructions (when to load/unload skills, "
            "workflow guidance) when any skill management tool is present."
        ),
    )


class System(BaseModel):
    """System message configuration for AI behavior and prompting."""

    use_default_prompt: bool = Field(
        default=False,
        description=(
            "Deprecated: legacy toggle for built-in default prompt injection. "
            "Use system.prompt to control per-category prompt injection."
        ),
        json_schema_extra={"deprecated": True},
    )
    text: str | None = Field(
        default=None,
        description="System prompt to use for the chat completion",
    )
    citations: Citations = Field(
        default=Citations(),
        description="Citation configuration for source attribution in responses",
    )
    extensions: list[SystemExtensions] = Field(
        default_factory=list,
        description="Set of enabled extensions",
    )
    blob_visibility: BlobVisibilityMode = Field(
        default=BlobVisibilityMode.PUBLIC,
        description="Controls how blobs are exposed: binary (raw data), internal (private S3 URI), or public (public S3 URL)",
    )
    prompt: PromptConfig = Field(
        default_factory=PromptConfig,
        description="Controls which platform-level prompt features are injected.",
    )

    @model_validator(mode="after")
    def _warn_use_default_prompt(self) -> "System":
        if self.use_default_prompt:
            warnings.warn(
                "System.use_default_prompt is deprecated and will be removed in a future version. "
                "Use system.prompt to control per-category prompt injection.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self


def validate_system_config(
    system: Sequence[System | TextBlock | str | dict[str, Any]]
    | System
    | TextBlock
    | str
    | dict[str, Any]
    | None,
) -> System:
    """Normalize system configuration (None, str, dict, list, or System)."""
    # None -> default empty System
    if system is None:
        return System()

    # If already a System instance, return as-is
    if isinstance(system, System):
        return system

    # TextBlock -> System(text=...)
    if isinstance(system, TextBlock):
        return System(text=system.text)

    # String -> System(text=...)
    if isinstance(system, str):
        return System(text=system)

    # Dict -> try to convert to System via pydantic
    if isinstance(system, dict):
        try:
            return System.model_validate(system)
        except Exception as e:
            raise ValueError(f"Invalid system specification (dict): {system}") from e

    # List: allow list of System / str / dict and convert+merge
    if isinstance(system, list):
        # Convert each item to System
        converted: list[System] = []
        for item in system:
            if isinstance(item, System):
                converted.append(item)
            elif isinstance(item, TextBlock):
                converted.append(System(text=item.text))
            elif isinstance(item, str):
                converted.append(System(text=item))
            elif isinstance(item, dict):
                try:
                    converted.append(System.model_validate(item))
                except Exception as e:
                    raise ValueError(
                        f"Invalid system item in list (dict): {item}"
                    ) from e
            else:
                raise ValueError(f"Invalid system item in list: {item}")

        if not converted:
            return System()

        # Merge converted System objects into a single System
        potential_system = converted[0]
        for item in converted[1:]:
            merged_text = None
            if potential_system.text or item.text:
                # concatenate texts with newline when both present
                if potential_system.text and item.text:
                    merged_text = f"{potential_system.text}\n{item.text}"
                else:
                    merged_text = potential_system.text or item.text

            potential_system = System(
                text=merged_text,
                use_default_prompt=item.use_default_prompt
                or potential_system.use_default_prompt,
                citations=Citations(
                    enabled=(
                        item.citations.enabled or potential_system.citations.enabled
                    ),
                    known_citations=list(
                        {
                            *(potential_system.citations.known_citations or []),
                            *(item.citations.known_citations or []),
                        }
                    ),
                ),
                extensions=list(
                    dict.fromkeys(
                        [*(potential_system.extensions or []), *(item.extensions or [])]
                    )
                ),
                blob_visibility=item.blob_visibility,
            )

        return potential_system

    # Unknown type
    raise ValueError(f"Invalid system specification: {system}")


SystemOrStr = Annotated[System, BeforeValidator(validate_system_config)]


class ToolChoice(BaseModel):
    """Configuration for tool selection behavior during AI interactions."""

    type: Literal["auto", "any", "tool", "none"] = Field(
        default="auto", description="Tool selection strategy"
    )
    name: str | None = Field(
        default=None, description="Name of the tool to use if not auto-selecting"
    )
    disable_parallel_tool_use: bool = Field(
        default=False,
        description="When true, prevents the AI from using multiple tools simultaneously",
    )
    validation_mode: Literal["eager", "lazy"] = Field(
        default="lazy",
        description=(
            "Tool validation mode. 'eager' validates tool calls before execution, "
            "'lazy' validates if tool call is made."
        ),
    )


class Thinking(BaseModel):
    """Configuration for AI reasoning and step-by-step thinking capabilities."""

    enabled: bool = Field(
        default=False,
        description="Enable reasoning capabilities for the model, allowing it to think step-by-step",
    )
    effort: Literal["low", "medium", "high", "max", "xhigh"] | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated. Use output_config.effort instead. "
            "Kept for backward compatibility with legacy clients."
        ),
    )


class ResponseFormatType(enum.StrEnum):
    """Enumeration of supported response formats."""

    text = "text"
    json_schema = "json_schema"


class ResponseFormat(BaseModel):
    """Deprecated response format model. Use JsonObjectFormat."""

    type: ResponseFormatType = Field(
        default=ResponseFormatType.text, description="Response format type"
    )
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema definition when type is 'json_schema'"
    )

    model_config = ConfigDict(json_schema_extra={"deprecated": True})


class JsonObjectFormat(BaseModel):
    """Structured JSON object format compatible with Anthropic output_config.format."""

    type: Literal["json_schema"] = Field(
        description='Output format type. Always "json_schema".',
    )
    json_schema: dict[str, Any] = Field(
        alias="schema",
        serialization_alias="schema",
        validation_alias=AliasChoices("schema", "json_schema"),
        description="JSON schema used to constrain the model output.",
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class OutputConfigInput(BaseModel):
    """Output configuration shared across Anthropic-compatible request models."""

    effort: Literal["low", "medium", "high", "max", "xhigh"] | None = Field(
        default=None,
        description="Reasoning effort level for output generation.",
    )
    format: JsonObjectFormat | None = Field(
        default=None,
        description="Optional structured output format schema.",
    )

    model_config = ConfigDict(extra="forbid")


class MessageInput(BaseModel):
    """Input message for AI conversations."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message sender"
    )
    content: str | list[Annotated[ContentBlockType, Field(discriminator="type")]] = (
        Field(description="The message content")
    )

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        json_schema_extra={"discriminator": {"propertyName": "role"}},
    )

    @classmethod
    def convert_from_llama_index_messages(
        cls,
        messages: Sequence["MessageInput"],
        converter: dict[str, ToolUseBlock] | None = None,
    ) -> list["ChatMessage"]:
        converter = converter or {}
        result = []

        messages = cls._support_legacy_messages(list(messages))
        messages = cls._merge_messages(messages)
        messages = cls._process_messages(messages)

        for msg in messages:
            llama_messages, converter = msg._convert_into_llama_index_messages(
                converter
            )
            result.extend(llama_messages)

        result = cls._reorder_tool_messages(result)
        result = cls._reorder_tldr_messages(result)
        cls._validate_message_order(result)

        return result

    @classmethod
    def _support_legacy_messages(
        cls, messages: list["MessageInput"]
    ) -> list["MessageInput"]:
        """Support legacy messages by converting them to the new format."""
        if not messages:
            return []

        converted_messages: list[MessageInput] = []
        for msg in messages:
            if msg.role == "user" and isinstance(msg.content, list):
                any_tool_result = any(
                    isinstance(block, ToolResultBlock) for block in msg.content
                )
                if any_tool_result:
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            # The current spec de Anthropic sends ToolResultBlock
                            # as a part of the user message.
                            # To support it, we need to convert it
                            converted_messages.append(
                                MessageInput(
                                    role="assistant",
                                    content=[block],
                                )
                            )
                        else:
                            converted_messages.append(
                                MessageInput(
                                    role=msg.role,
                                    content=[block],
                                )
                            )
                else:
                    converted_messages.append(
                        MessageInput(
                            role=msg.role,
                            content=msg.content,
                        )
                    )
            else:
                # If the message is a user message with content as a list,
                # we assume it's already in the correct format.
                converted_messages.append(
                    MessageInput(
                        role=msg.role,
                        content=msg.content,
                    )
                )

        return converted_messages

    @classmethod
    def _process_messages(cls, messages: list["MessageInput"]) -> list["MessageInput"]:
        result: list[MessageInput] = []

        for msg in messages:
            if msg.role == "assistant":
                current_blocks: list[ContentBlockType] = []
                for block in msg.content or []:
                    if isinstance(block, ToolUseBlock):
                        result.append(
                            MessageInput(
                                role="assistant",
                                content=[*current_blocks, block],
                            )
                        )
                        current_blocks = []
                    elif isinstance(block, ToolResultBlock):
                        if current_blocks:
                            result.append(
                                MessageInput(
                                    role=msg.role,
                                    content=current_blocks,
                                )
                            )
                            current_blocks = []

                        result.append(
                            MessageInput(
                                role="assistant",
                                content=[block],
                            )
                        )
                    elif isinstance(block, ContentBlockType):
                        current_blocks.append(block)
                    elif isinstance(block, str):
                        current_blocks.append(TextBlock(text=block))
                    else:
                        raise ValueError(
                            f"Unsupported content block type: {type(block)}"
                        )

                if current_blocks:
                    result.append(
                        MessageInput(
                            role=msg.role,
                            content=current_blocks,
                        )
                    )
            else:
                result.append(msg)

        return result

    @classmethod
    def _validate_message_order(cls, messages: list[ChatMessage]) -> None:
        """Validates the order of the messages."""
        previous_role: MessageRole | None = None

        for message in messages:
            current_role = message.role

            if previous_role is not None:
                expected_roles: set[MessageRole] = set()

                if previous_role == MessageRole.SYSTEM:
                    expected_roles = {
                        MessageRole.USER,
                        MessageRole.ASSISTANT,
                    }
                elif previous_role == MessageRole.USER:
                    expected_roles = {
                        MessageRole.ASSISTANT,
                        MessageRole.USER,
                    }
                elif previous_role == MessageRole.ASSISTANT:
                    expected_roles = {
                        MessageRole.USER,
                        MessageRole.TOOL,
                    }
                elif previous_role == MessageRole.TOOL:
                    expected_roles = {
                        MessageRole.ASSISTANT,
                        # Mistral doesn't support user after tool
                        # Fixed in the tokenizer
                        MessageRole.USER,
                    }

                if current_role not in expected_roles:
                    raise ValueError(
                        f"Invalid message order: expected {expected_roles} after {previous_role}, but got {current_role}."
                        "Check ToolUseBlock and ToolResultBlock order."
                        if current_role == MessageRole.TOOL
                        else ""
                    )

            previous_role = current_role

    @classmethod
    def _merge_messages(cls, messages: list["MessageInput"]) -> list["MessageInput"]:
        """Merge consecutive messages with the same role."""
        if not messages:
            return []

        merged_messages: list[MessageInput] = []
        current_message = messages[0]

        for msg in messages[1:]:
            if msg.role == current_message.role:
                merged_content = cls._merge_content(
                    current_message.content, msg.content
                )
                current_message = MessageInput(
                    role=current_message.role, content=merged_content
                )
            else:
                merged_messages.append(current_message)
                current_message = msg

        merged_messages.append(current_message)
        return merged_messages

    @classmethod
    def _merge_content(
        cls,
        content1: str | list[ContentBlockType],
        content2: str | list[ContentBlockType],
    ) -> str | list[ContentBlockType]:
        """Merge two content objects handling different types."""
        # Both string: concatenate with newline
        if isinstance(content1, str) and isinstance(content2, str):
            return f"{content1}\n{content2}"

        # Convert strings to TextBlock lists for consistent handling
        blocks1 = cls._normalize_to_blocks(content1)
        blocks2 = cls._normalize_to_blocks(content2)

        return blocks1 + blocks2

    @classmethod
    def _normalize_to_blocks(
        cls, content: str | list[ContentBlockType] | None
    ) -> list[ContentBlockType]:
        """Convert content to list of ContentBlockType."""
        if content is None:
            return []

        if isinstance(content, str):
            return [TextBlock(text=content)]

        return content

    @classmethod
    def _flat_tool_messages(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Flatten tool messages to ensure they are in the correct order."""
        flat_messages = []
        for message in messages:
            if "tool_calls" in message.additional_kwargs:
                tool_calls: list[ToolSelection] = message.additional_kwargs[
                    "tool_calls"
                ]
                for i, tool_call in enumerate(tool_calls):
                    copy_message = message.model_copy(deep=True)
                    if i != 0:
                        copy_message.blocks = []
                    copy_message.additional_kwargs["tool_calls"] = [tool_call]
                    flat_messages.append(copy_message)
            else:
                flat_messages.append(message)

        return flat_messages

    @classmethod
    def _reorder_tool_messages(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure tool responses immediately follow their assistant messages."""
        result = []
        used_indices = set()

        for i, message in enumerate(messages):
            if i in used_indices:
                continue

            if "tldr" in message.additional_kwargs:
                # Skip TLDR messages, they are handled separately
                result.append(message)
                used_indices.add(i)

            elif (
                message.role == MessageRole.ASSISTANT
                and message.additional_kwargs.get("tool_calls")
            ):
                tool_call_ids = {
                    tool_call.tool_id
                    for tool_call in message.additional_kwargs["tool_calls"]
                }

                tool_responses = []
                for j, msg in enumerate(messages):
                    if (
                        j != i
                        and msg.role == MessageRole.TOOL
                        and msg.additional_kwargs.get("tool_call_id") in tool_call_ids
                    ):
                        tool_responses.append(msg)
                        used_indices.add(j)

                result.append(message)
                result.extend(tool_responses)
                used_indices.add(i)

            elif message.role == MessageRole.TOOL:
                # Skip tools that can be before the assistant message
                pass
            else:
                result.append(message)
                used_indices.add(i)

        return result

    @classmethod
    def _reorder_tldr_messages(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Reorder TLDR messages before their respective user messages.

        Left TLDRs (tldr_side="left") represent condensed conversation history and
        must be placed before their anchor user message, clearing all prior history.
        Right TLDRs (tldr_side="right") are tool-call summaries already in the correct
        position after the user message — no reordering needed.
        """
        if not messages:
            return messages

        def _has_tldr_content(message: ChatMessage) -> bool:
            tldr_content = message.additional_kwargs.get("tldr", [])
            return bool(tldr_content)

        def _is_left_tldr(message: ChatMessage) -> bool:
            return message.additional_kwargs.get("tldr") == "left"

        # Split into blocks by user messages
        user_blocks = cls._split_messages(
            messages,
            lambda msg: msg.role == MessageRole.USER and not _has_tldr_content(msg),
        )

        if not user_blocks:
            return messages

        result: list[ChatMessage] = []
        # Track by object identity (id) rather than value equality.
        # Using value equality would incorrectly deduplicate distinct ChatMessage
        # instances that happen to have the same content — e.g. multiple TOOL("-")
        # entries produced by accumulated right-side TLDRBlocks.
        result_ids: set[int] = set()

        def add_by_id(msgs: list[ChatMessage]) -> None:
            for msg in msgs:
                if id(msg) not in result_ids:
                    result.append(msg)
                    result_ids.add(id(msg))

        for block in user_blocks:
            if not block:
                continue

            # Find TLDR messages and their positions in this block
            tldr_positions: list[int] = []
            for j, message in enumerate(block):
                if _has_tldr_content(message):
                    tldr_positions.append(j)

            if not tldr_positions:
                # No TLDR in this block, keep all messages as-is
                add_by_id(block)
                continue

            user_message: ChatMessage = block[0]
            if user_message and user_message.role != MessageRole.USER:
                raise ValueError("First message in the block must be a USER message.")

            # Group consecutive TLDR positions together so we can detect accumulated
            # TLDRBlocks (where each subsequent block repeats all prior entries plus
            # new ones at the end).
            tldr_positions_grouped_by_consecutive = [
                [num for _, num in group]
                for _, group in groupby(
                    enumerate(tldr_positions), lambda x: x[1] - x[0]
                )
            ]

            # Tracks how many TLDR ChatMessages were emitted by the previous group.
            # Accumulated right-side TLDRBlocks re-emit all prior summaries as a
            # prefix, so we skip that prefix to avoid duplicates while still
            # preserving distinct instances with identical text (e.g. TOOL("-")).
            prev_group_tldr_count = 0

            for i, pos in enumerate(tldr_positions_grouped_by_consecutive):
                earliest_tldr_pos = min(pos)
                latest_tldr_pos = max(pos) + 1

                if i == len(tldr_positions_grouped_by_consecutive) - 1:
                    # Last group: include everything up to the end of the block
                    latest_tldr_pos = len(block)

                remaining_messages = block[earliest_tldr_pos:latest_tldr_pos]
                block_user_message: list[ChatMessage] = [user_message]

                # Check if this TLDR group contains left-side TLDRs.
                # Left TLDRs represent condensed history and must be placed before
                # the user message, clearing all prior history.
                # Right TLDRs are tool-call summaries already in the correct position.
                has_left_tldr = any(
                    msg.role == MessageRole.USER and _is_left_tldr(msg)
                    for msg in remaining_messages
                )

                if not has_left_tldr:
                    # Right TLDRs: order is already correct,
                    # just ensure user comes first
                    add_by_id(block_user_message)

                    tldr_in_remaining = [
                        m for m in remaining_messages if _has_tldr_content(m)
                    ]

                    # Detect accumulated TLDRBlocks: each new block contains all entries
                    # from the previous block as a prefix, plus new entries appended at
                    # the end. We skip the prefix by count (not by value equality) so
                    # that distinct instances with identical text —
                    # such as multiple TOOL("-") entries
                    # — are not incorrectly collapsed into one.
                    if len(tldr_in_remaining) > prev_group_tldr_count:
                        # This block has more TLDR entries than the previous one,
                        # meaning the first `prev_group_tldr_count`
                        # are the repeated prefix.
                        skip_count = prev_group_tldr_count
                    else:
                        # Independent (non-accumulated) TLDR: emit all entries fresh.
                        skip_count = 0

                    prev_group_tldr_count = len(tldr_in_remaining)

                    skipped = 0
                    for m in remaining_messages:
                        if _has_tldr_content(m):
                            if skipped < skip_count:
                                # Skip this entry — it was already emitted by the
                                # previous accumulated TLDR group.
                                skipped += 1
                            else:
                                result.append(m)
                                result_ids.add(id(m))
                        else:
                            add_by_id([m])
                else:
                    # Left TLDRs: separate TLDR and non-TLDR from remaining messages
                    block_tldr_messages: list[ChatMessage] = []
                    block_non_tldr_messages: list[ChatMessage] = []

                    for message in remaining_messages:
                        if _has_tldr_content(message):
                            block_tldr_messages.append(message)
                        else:
                            block_non_tldr_messages.append(message)

                    # As we found a left TLDR, we can remove all prior history
                    result.clear()
                    result_ids.clear()
                    # Reset accumulated count since history was cleared
                    prev_group_tldr_count = 0

                    # Add TLDR messages first, then user message and non-TLDR messages
                    add_by_id(block_tldr_messages)
                    add_by_id(block_user_message)
                    add_by_id(block_non_tldr_messages)

        # Remove consecutive same-role messages caused by TLDR reordering
        last_message: ChatMessage | None = None
        final_result: list[ChatMessage] = []
        final_result_ids: set[int] = set()

        for message in result:
            if last_message and last_message.role == message.role:
                if "tldr" in message.additional_kwargs:
                    if id(last_message) in final_result_ids:
                        final_result.remove(last_message)
                        final_result_ids.discard(id(last_message))

                if "tldr" in last_message.additional_kwargs:
                    if id(last_message) not in final_result_ids:
                        final_result.append(last_message)
                        final_result_ids.add(id(last_message))
                else:
                    final_result.append(message)
                    final_result_ids.add(id(message))
            else:
                final_result.append(message)
                final_result_ids.add(id(message))

            last_message = message

        return final_result

    @classmethod
    def _split_messages(
        cls, messages: list[ChatMessage], split_condition: Callable[[ChatMessage], bool]
    ) -> list[list[ChatMessage]]:
        """Split messages into groups based on condition."""
        groups: list[list[ChatMessage]] = []
        current_group: list[ChatMessage] = []

        for msg in messages:
            if split_condition(msg) and current_group:
                groups.append(current_group)
                current_group = [msg]
            else:
                current_group.append(msg)

        if current_group:
            groups.append(current_group)

        return groups

    def _convert_into_llama_index_messages(
        self,
        tool_uses: dict[str, ToolUseBlock] | None = None,
    ) -> tuple[list["ChatMessage"], dict[str, ToolUseBlock]]:
        tool_uses = tool_uses or {}

        if self.role in ["system", "user"]:
            return self._convert_message(), tool_uses
        elif self.role == "assistant":
            return self._convert_assistant_message(tool_uses)
        else:
            raise ValueError(
                f"Unknown message role {self.role}. Expected 'system', 'user', 'assistant', or 'tool'."
            )

    def _extract_content(
        self, content: str | list[ContentBlockType] | None
    ) -> tuple[list[LIContentBlock] | None, dict[str, list[ContentBlockType]] | None]:
        """Extract text content from various content types."""
        blocks: list[LIContentBlock] = []
        custom_blocks: dict[str, list[ContentBlockType]] = {}
        if isinstance(content, str):
            blocks.append(LITextBlock(text=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, TextBlock):
                    blocks.append(LITextBlock(text=block.text))
                elif isinstance(block, MidConvSystemBlock):
                    text = "\n".join(b.text for b in block.content)
                    blocks.append(LITextBlock(text=text))
                elif isinstance(block, ImageBlock):
                    image_bytes = load_file_from_uri(block.source.get_data())
                    image_size = len(image_bytes.read())
                    if image_size > settings().chat.maximum_blob_size:
                        raise ValueError(
                            f"Image size {image_size} exceeds maximum "
                            f"allowed size of {settings().chat.maximum_blob_size} bytes."
                        )

                    image_bytes.seek(0)
                    blocks.append(
                        LIImageBlock(
                            image=image_bytes.read(),
                            image_mimetype=block.source.get_media_type(),
                        )
                    )
                elif isinstance(block, AudioBlock):
                    audio_bytes = load_file_from_uri(block.source.get_data())
                    audio_size = len(audio_bytes.read())
                    if audio_size > settings().chat.maximum_blob_size:
                        raise ValueError(
                            f"Audio size {audio_size} exceeds maximum "
                            f"allowed size of {settings().chat.maximum_blob_size} bytes."
                        )
                    audio_bytes.seek(0)
                    blocks.append(
                        LIAudioBlock(
                            audio=audio_bytes.read(),
                            format=block.source.get_media_type(),
                        )
                    )
                elif isinstance(block, ContentBlockType):
                    if block.type not in custom_blocks:
                        custom_blocks[block.type] = []
                    custom_blocks[block.type].append(block)

        return blocks if blocks else None, custom_blocks if custom_blocks else None

    def _convert_message(self) -> list[ChatMessage]:
        """Convert system or user messages."""
        li_blocks, additional_kwargs = self._extract_content(self.content)
        return [
            ChatMessage(
                role=MessageRole(self.role),
                content=li_blocks,
                additional_kwargs=additional_kwargs or {},
            )
        ]

    def _convert_assistant_message(
        self, tool_uses: dict[str, ToolUseBlock]
    ) -> tuple[list[ChatMessage], dict[str, ToolUseBlock]]:
        """Convert assistant messages with potential tool blocks."""
        if not isinstance(self.content, list):
            content = str(self.content) if self.content else None
            return [ChatMessage(role=MessageRole.ASSISTANT, content=content)], tool_uses

        messages: list[ChatMessage] = []
        current_blocks: list[ContentBlockType] = []
        current_additional_kwargs: dict[str, Any] = {}

        # Convert content blocks to LlamaIndex blocks
        for block in self.content:
            if isinstance(block, ToolUseBlock):
                if block:
                    if "tool_calls" not in current_additional_kwargs:
                        current_additional_kwargs["tool_calls"] = []

                    current_additional_kwargs["tool_calls"].append(
                        ToolSelection(
                            tool_id=block.id,
                            tool_name=block.name,
                            tool_kwargs=block.input,
                        )
                    )
                    tool_uses[block.id] = block

            elif isinstance(block, ToolResultBlock):
                if current_blocks:
                    li_blocks, custom_blocks = self._extract_content(current_blocks)
                    messages.append(
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=li_blocks,
                            additional_kwargs=current_additional_kwargs,
                        )
                    )

                    current_blocks = []
                    current_additional_kwargs = {}

                li_blocks, custom_blocks = self._extract_content(block.content)  # type: ignore
                tool_use = tool_uses.get(block.tool_use_id)

                if li_blocks is None and not custom_blocks:
                    li_blocks = [LITextBlock(text="No content")]

                additional_kwargs: dict[str, Any] = {
                    **(custom_blocks or {}),
                    "tool_call_id": block.tool_use_id,
                    "tool_call_name": tool_use.name if tool_use else None,
                    "tool_call_args": tool_use.input if tool_use else None,
                }

                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=li_blocks,
                        additional_kwargs=additional_kwargs,
                    )
                )

            elif isinstance(block, TLDRBlock):
                if current_blocks or current_additional_kwargs:
                    li_blocks, custom_blocks = self._extract_content(current_blocks)
                    messages.append(
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=li_blocks,
                            additional_kwargs=current_additional_kwargs,
                        )
                    )
                current_blocks = []
                current_additional_kwargs = {}

                # Only create TLDR messages if content is not empty
                if block.content and isinstance(block.content, list):
                    for ct in block.content:
                        msg_role: str | None = None
                        if isinstance(ct, BaseContentBlock):
                            msg_role = ct.metadata.get("role")
                        if msg_role and isinstance(ct, TextBlock):
                            li_blocks, custom_blocks = self._extract_content([ct])
                            messages.append(
                                ChatMessage(
                                    role=MessageRole(msg_role),
                                    content=li_blocks,
                                    additional_kwargs={
                                        **(custom_blocks or {}),
                                        # Preserve the tldr_side from the block so that
                                        # _reorder_tldr_messages can distinguish left
                                        # (history condensation) from right.
                                        "tldr": block.tldr_side,
                                    },
                                )
                            )
            else:
                current_blocks.append(block)

        # Handle any remaining blocks after the loop
        if current_blocks or current_additional_kwargs:
            li_blocks, custom_blocks = self._extract_content(current_blocks)
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=li_blocks,
                    additional_kwargs={
                        **(custom_blocks or {}),
                        **current_additional_kwargs,
                    },
                )
            )

        # Flatten tool messages to ensure they are in the correct order
        messages = self._flat_tool_messages(messages)

        return messages, tool_uses


class ToolSpecBody(BaseModel):
    """Definition for a tool the client can call."""

    name: str = Field(description="Unique name identifier for the tool")
    type: str | None = Field(
        default=None,
        description="Type of the tool, use to identify internal tools database_query_v1 or semantic_search_v1",
    )
    description: str | None = Field(
        default=None, description="Human-readable description of what the tool does"
    )
    input_schema: dict[str, Any] | None = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON schema defining the input parameters the tool accepts",
    )
    context: list[ArtifactType] | None = Field(
        default=None,
        description="Additional context or metadata for the tool",
    )
    defer_loading: bool = Field(
        default=False,
        description=(
            "When true, hide this tool from the model until at least one skill is "
            "loaded in the current conversation."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "Optional instructions injected into the system prompt when this tool "
            "is available. For internal tools a default template is used; providing "
            "a value here overrides that default. Set to an empty string to disable."
        ),
    )

    class Config:
        extra = "allow"
        alias_generator = to_camel
        populate_by_name = True


def validate_tool_spec(tool: ToolSpecBody | dict[str, Any] | str | Any) -> ToolSpecBody:
    """Convert string to ToolSpec or pass through ToolSpec objects."""
    # Internal tools have their description, and actual function
    # body set dynamically from the context, in the meantime
    # a placeholder tool is created to be replaced later with the actual tool
    if isinstance(tool, ToolSpecBody):
        return tool

    tool_type: str | None = None
    if isinstance(tool, dict):
        tool_type = tool.get("type") or "custom"
    elif isinstance(tool, str):
        tool_type = tool
    else:
        tool_type = "custom"

    if not isinstance(tool_type, str):
        tool_type = "custom"

    internal_tool_name = resolve_internal_tool_name(tool_type)
    if internal_tool_name is not None:
        # Tools baked in private-gpt
        tool_context: list[ArtifactType] | None = None
        if isinstance(tool, dict):
            tool_context = tool.get("context")
            if not isinstance(tool_context, list):
                tool_context = None
        # Try to get the custom name
        name: str = ""
        if isinstance(tool, dict) and tool.get("name"):
            name = tool.get("name") or ""

        return ToolSpecBody(
            name=name or internal_tool_name,
            type=tool_type,
            context=tool_context,
        )
    else:
        # Tools provided (and executed) by the caller
        try:
            return ToolSpecBody.model_validate(tool)
        except Exception as e:
            raise ValueError(f"Invalid tool specification: {tool}") from e


ToolSpecOrString = Annotated[ToolSpecBody, BeforeValidator(validate_tool_spec)]


class CompletionMetadata(BaseModel):
    user_id: str | None = Field(
        default=None,
        description="Opaque user identifier for request attribution.",
        max_length=512,
    )

    model_config = ConfigDict(extra="allow")


class MessagesInputBase(BaseModel):
    """Shared Anthropic-compatible input shape for message-based endpoints."""

    model: str = Field(default="default", description="Model identifier or alias.")
    messages: list[MessageInput] = Field(
        description="Conversation messages for the request."
    )
    system: list[System] = Field(
        default_factory=lambda: [System()],
        description=(
            "System prompt input. Accepts str, list[str], System, list[System], or null. "
            "It is normalized internally to list[System]."
        ),
    )
    tools: list[ToolSpecOrString] | None = Field(
        default=None,
        description="Optional tool definitions.",
    )
    thinking: Thinking = Field(
        default=Thinking(),
        description="Thinking configuration.",
    )
    tool_choice: ToolChoice = Field(
        default=ToolChoice(),
        description="Tool selection policy.",
        validation_alias=AliasChoices("tool_choice", "toolChoice"),
    )
    output_config: OutputConfigInput = Field(
        default_factory=OutputConfigInput,
        description="Optional output configuration options.",
        validation_alias=AliasChoices("output_config", "outputConfig"),
    )
    cache_control: (
        Annotated[CacheControlEphemeral, Field(discriminator="type")] | None
    ) = Field(
        default=None,
        description="Optional request-level cache control.",
        validation_alias=AliasChoices("cache_control", "cacheControl"),
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        schema = handler(core_schema)
        if isinstance(schema, dict):
            required = schema.get("required")
            if not isinstance(required, list):
                required = []
            for field in ("model", "messages"):
                if field not in required:
                    required.append(field)
            schema["required"] = sorted(required)
        return schema

    @field_validator("system", mode="before")
    @classmethod
    def normalize_system(
        cls,
        value: list[System | TextBlock | str | dict[str, Any]]
        | System
        | TextBlock
        | str
        | dict[str, Any]
        | None,
    ) -> list[System]:
        if value is None:
            return [System()]
        if isinstance(value, System):
            return [value]
        if isinstance(value, TextBlock):
            return [System(text=value.text)]
        if isinstance(value, str):
            return [System(text=value)]
        if isinstance(value, dict):
            if value.get("type") == "text" and isinstance(value.get("text"), str):
                return [System(text=value["text"])]
            return [System.model_validate(value)]
        if isinstance(value, list):
            systems: list[System] = []
            for item in value:
                if isinstance(item, System):
                    systems.append(item)
                elif isinstance(item, TextBlock):
                    systems.append(System(text=item.text))
                elif isinstance(item, str):
                    systems.append(System(text=item))
                elif isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        systems.append(System(text=item["text"]))
                    else:
                        systems.append(System.model_validate(item))
                else:
                    raise ValueError(f"Invalid system item: {item}")
            return systems or [System()]
        raise ValueError(f"Invalid system value: {value}")

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: str | None) -> str:
        if value is None:
            return "default"
        return value

    @model_validator(mode="after")
    def extract_system_messages(self) -> "MessagesInputBase":
        """Extract role=system messages and append them to the system list."""
        system_msgs = [msg for msg in self.messages if msg.role == "system"]
        if not system_msgs:
            return self

        self.messages = [msg for msg in self.messages if msg.role != "system"]

        for msg in system_msgs:
            if isinstance(msg.content, str):
                self.system.append(System(text=msg.content))
            elif isinstance(msg.content, list):
                texts = [
                    b.text for b in msg.content if isinstance(b, TextBlock) and b.text
                ]
                if texts:
                    self.system.append(System(text="\n".join(texts)))

        return self


class CompletionInput(BaseModel):
    """Anthropic completion request payload."""

    model: str = Field(description="Model identifier or alias.")
    prompt: str = Field(
        min_length=1, description="Legacy completion prompt in Human/Assistant format."
    )
    max_tokens_to_sample: Annotated[int, Ge(1)] = Field(
        description="Maximum number of tokens to sample.",
        validation_alias=AliasChoices("max_tokens_to_sample", "maxTokensToSample"),
    )
    metadata: CompletionMetadata | None = Field(
        default=None,
        description="Metadata object for request attribution.",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Stop generation if any sequence is encountered.",
        validation_alias=AliasChoices("stop_sequences", "stopSequences"),
    )
    stream: bool = Field(default=False, description="Whether to stream the response.")
    temperature: Annotated[float, Ge(0), Le(1)] | None = Field(
        default=None,
        description="Sampling temperature between 0 and 1.",
    )
    top_k: Annotated[int, Ge(0)] | None = Field(
        default=None,
        description="Top-k sampling parameter.",
        validation_alias=AliasChoices("top_k", "topK"),
    )
    top_p: Annotated[float, Ge(0), Le(1)] | None = Field(
        default=None,
        description="Top-p sampling parameter.",
        validation_alias=AliasChoices("top_p", "topP"),
    )

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, value: str | None) -> str:
        if value is None:
            return "default"
        return value


class CompletionOutput(BaseModel):
    """Anthropic completion response payload."""

    id: str = Field(description="Completion identifier")
    type: Literal["completion"] = Field(description='Object type. Always "completion".')
    completion: str = Field(description="Generated completion text.")
    stop_reason: str | None = Field(description="Reason the generation stopped.")
    model: str = Field(description="Resolved model identifier.")


class CapabilitySupportOutput(BaseModel):
    supported: bool = Field(description="Whether a capability is supported.")


class CountCapabilitySupportOutput(CapabilitySupportOutput):
    maximum: int = Field(default=0, description="Maximum number of allowed elements.")


class ThinkingTypesOutput(BaseModel):
    adaptive: CapabilitySupportOutput = Field(
        description='Support for thinking type "adaptive".'
    )
    enabled: CapabilitySupportOutput = Field(
        description='Support for thinking type "enabled".'
    )


class ThinkingCapabilityOutput(BaseModel):
    supported: bool = Field(description="Whether thinking is supported.")
    types: ThinkingTypesOutput = Field(
        description="Thinking type support configuration."
    )


class EffortCapabilityOutput(BaseModel):
    supported: bool = Field(description="Whether effort control is supported.")
    low: CapabilitySupportOutput = Field(description='Support for effort "low".')
    medium: CapabilitySupportOutput = Field(description='Support for effort "medium".')
    high: CapabilitySupportOutput = Field(description='Support for effort "high".')
    max: CapabilitySupportOutput = Field(description='Support for effort "max".')
    xhigh: CapabilitySupportOutput = Field(description='Support for effort "xhigh".')


class ContextManagementCapabilityOutput(BaseModel):
    clear_thinking_20251015: CapabilitySupportOutput | None = Field(
        description='Support for strategy "clear_thinking_20251015".'
    )
    clear_tool_uses_20250919: CapabilitySupportOutput | None = Field(
        description='Support for strategy "clear_tool_uses_20250919".'
    )
    compact_20260112: CapabilitySupportOutput | None = Field(
        description='Support for strategy "compact_20260112".'
    )
    supported: bool = Field(description="Whether context management is supported.")


class ModelCapabilitiesOutput(BaseModel):
    batch: CapabilitySupportOutput = Field(description="Batch API support.")
    citations: CapabilitySupportOutput = Field(description="Citation support.")
    code_execution: CapabilitySupportOutput = Field(
        description="Code execution support."
    )
    context_management: ContextManagementCapabilityOutput = Field(
        description="Context management capabilities."
    )
    effort: EffortCapabilityOutput = Field(description="Reasoning effort capabilities.")
    image_input: CountCapabilitySupportOutput = Field(
        description="Image input support."
    )
    audio_input: CountCapabilitySupportOutput | None = Field(
        default=None, description="Audio input support."
    )
    pdf_input: CapabilitySupportOutput = Field(description="PDF input support.")
    structured_outputs: CapabilitySupportOutput = Field(
        description="Structured output support."
    )
    thinking: ThinkingCapabilityOutput = Field(description="Thinking support.")


class ModelInfoOutput(BaseModel):
    """Model metadata payload."""

    id: str = Field(description="Unique model identifier.")
    created_at: datetime = Field(description="Model release timestamp (RFC3339).")
    display_name: str = Field(description="Human-readable model name.")
    type: Literal["model"] = Field(description='Object type "model".')
    max_tokens: int | None = Field(
        description="Maximum value allowed for max_tokens for this model."
    )
    max_input_tokens: int | None = Field(
        description="Maximum input context window for this model."
    )
    embed_dim: int | None = Field(
        default=None,
        description="Embedding vector dimension for embedding models.",
    )
    capabilities: ModelCapabilitiesOutput | None = Field(
        description="Detailed model capability map."
    )


class ModelListOutput(BaseModel):
    """Paginated model list payload."""

    data: list[ModelInfoOutput] = Field(description="List of model objects.")
    first_id: str | None = Field(
        description="First model id in page, usable as before_id."
    )
    has_more: bool = Field(description="Whether more models are available.")
    last_id: str | None = Field(
        description="Last model id in page, usable as after_id."
    )


class CountTokensInput(MessagesInputBase):
    """Anthropic /messages/count_tokens payload."""

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "Count tokens for this input.",
                    }
                ],
                "system": [
                    {
                        "text": "You are a tokenizer.",
                    }
                ],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name.",
                                }
                            },
                            "required": ["city"],
                        },
                    }
                ],
                "tool_choice": {
                    "type": "auto",
                    "disable_parallel_tool_use": False,
                    "validation_mode": "lazy",
                },
                "thinking": {"enabled": False},
            }
        },
    )


class CountTokensOutput(BaseModel):
    """Token count payload."""

    input_tokens: int = Field(
        description="Estimated number of input tokens for the provided payload."
    )
