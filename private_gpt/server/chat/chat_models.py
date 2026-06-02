from typing import Annotated, Any, ClassVar, Literal

from annotated_types import Ge, Le
from llama_index.core.base.llms.types import ChatMessage
from pydantic import ConfigDict, Field, WithJsonSchema, model_validator

from private_gpt.chat.input_models import (
    CompletionMetadata,
    MessageInput,
    MessagesInputBase,
    ResponseFormat,
    ResponseFormatType,
    System,
    validate_system_config,
)
from private_gpt.events.models import TLDRBlock, ToolResultBlock, ToolUseBlock
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.utils.artifact_input import ArtifactType


class ChatBody(MessagesInputBase):
    """Chat request body model for handling chat interactions."""

    stream: bool = Field(
        default=False,
        description="Whether to stream the response back to the client.",
    )
    tool_context: list[ArtifactType] | None = Field(
        default=None,
        description="""Context to provide to the tools, such as documents,
        databases connection strings, or data relevant to tool usage.""",
    )
    mcp_servers: list[McpServerConfig] = Field(
        default_factory=list,
        description="""List of MCP servers to use for tool retrieval. Each server can have its own configuration.""",
    )
    container: str | None = Field(
        default=None,
        description="Container identifier for reuse across requests.",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat(),
        description="""Deprecated response format. Use output_config.format instead.""",
    )
    priority: int | None = Field(
        default=None,
        description="""Priority of the request, used for prioritizing responses.""",
    )
    seed: int | None = Field(
        default=None,
        description="""Random seed for reproducibility.""",
    )
    min_p: float | None = Field(
        default=None,
        description="""Minimum probability threshold for token selection. Tokens with probability below this value are filtered out.""",
    )
    top_p: Annotated[
        float | None,
        Ge(0),
        Le(1),
        WithJsonSchema({"type": "number", "minimum": 0, "maximum": 1}),
    ] = Field(
        default=None,
        description="""Nucleus sampling parameter. Only tokens with cumulative probability up to this value are considered.""",
    )
    temperature: Annotated[
        float | None,
        Ge(0),
        Le(1),
        WithJsonSchema({"type": "number", "minimum": 0, "maximum": 1}),
    ] = Field(
        default=None,
        description="""Controls randomness in generation. Higher values make output more random, lower values more deterministic.""",
    )
    top_k: Annotated[
        int | None,
        Ge(0),
        WithJsonSchema({"type": "integer", "minimum": 0}),
    ] = Field(
        default=None,
        description="""Limits token selection to the top K most likely tokens at each step.""",
    )
    repetition_penalty: float | None = Field(
        default=None,
        description="""Penalty applied to tokens that have already appeared in the sequence to reduce repetition.""",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="""Penalty applied based on whether a token has appeared in the text, encouraging topic diversity.""",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="""Penalty applied based on how frequently a token appears in the text, reducing repetitive content.""",
    )
    max_tokens: Annotated[
        int | None,
        WithJsonSchema({"type": "integer", "minimum": 1}),
    ] = Field(
        default=None,
        description="""Maximum number of tokens to generate in the response.""",
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="Custom stop sequences that stop generation when matched.",
    )
    metadata: CompletionMetadata = Field(
        default_factory=CompletionMetadata,
        description="Request metadata (for example, user_id).",
    )
    service_tier: Literal["auto", "standard_only"] = Field(
        default="auto",
        description='Service tier preference (for example, "auto" or "standard_only").',
    )
    inference_geo: str | None = Field(
        default=None,
        description="Geographic region hint for inference processing.",
    )
    correlation_id: str | None = Field(
        default=None,
        description="""Correlation ID for tracking the request across systems.""",
    )
    maximum_loaded_skills: int | None = Field(
        default=None,
        description=(
            "Optional cap for concurrently loaded skills in a conversation. "
            "When exceeded, the oldest loaded skill is evicted."
        ),
        ge=1,
    )
    context_management: Any | None = Field(
        default=None,
        description="Optional context management configuration",
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "How do you fry an egg? Choose the best method.",
                        },
                    ],
                    "stream": False,
                    "tools": [
                        {
                            "name": "egg_fryer",
                            "description": "A tool to fry eggs with precise temperature control",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "temperature": {
                                        "type": "number",
                                        "description": "Temperature in degrees Celsius",
                                    },
                                    "time": {
                                        "type": "number",
                                        "description": "Time in minutes to fry the egg",
                                    },
                                },
                                "required": ["temperature", "time"],
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "auto",
                        "disable_parallel_tool_use": False,
                    },
                    "response_format": {"type": "text"},
                    "system": {
                        "text": "You are a helpful cooking assistant. Provide clear, step-by-step instructions.",
                        "citations": {"enabled": True},
                    },
                    "thinking": {"enabled": False},
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What's the weather like today?",
                        },
                    ],
                    "stream": True,
                    "mcp_servers": [
                        {
                            "url": "http://localhost:8080/mcp",
                            "tool_configuration": {
                                "enabled": True,
                                "enabled_tools": ["weather_get", "weather_forecast"],
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "auto",
                    },
                    "system": {
                        "text": "You are a weather assistant. Provide current and accurate weather information.",
                        "citations": {"enabled": False},
                    },
                    "thinking": {"enabled": True},
                    "top_p": 0.9,
                    "temperature": 0.3,
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Generate a JSON response with user profile data",
                        },
                    ],
                    "stream": False,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "number"},
                            },
                        },
                    },
                    "system": {
                        "text": "You are a data generator. Always respond with valid JSON.",
                        "citations": {"enabled": False},
                    },
                    "thinking": {"enabled": False},
                    "tool_choice": {
                        "type": "none",
                    },
                    "seed": 42,
                },
                {
                    "messages": [
                        {
                            "content": "How many users are there in the users table?",
                            "role": "user",
                        }
                    ],
                    "tool_context": [
                        {
                            "type": "sql_database",
                            "connection_string": "postgres://postgres:postgres@localhost:5432/main",
                            "schemas": ["public"],
                        }
                    ],
                    "tools": [{"name": "database_query", "type": "database_query_v1"}],
                },
            ]
        }
    )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Any:
        schema = handler(core_schema)
        if isinstance(schema, dict):
            required = schema.get("required")
            if not isinstance(required, list):
                required = []
            for field in ("model", "messages", "max_tokens"):
                if field not in required:
                    required.append(field)
            schema["required"] = sorted(required)
        return schema

    _valid_last_message_roles: ClassVar[list[str]] = ["user", "assistant"]

    def llama_index_messages(self) -> list[ChatMessage]:
        """Convert messages to LlamaIndex format."""
        return MessageInput.convert_from_llama_index_messages(self.messages)

    def system_list(self) -> list[System]:
        """Return system configuration normalized as a list of System blocks."""
        return self.system

    def merged_system(self) -> System:
        """Return system configuration merged into a single System object."""
        return validate_system_config(self.system_list())

    @model_validator(mode="after")
    def validate_properties(self) -> "ChatBody":
        system = self.merged_system()

        if self.top_p and self.top_p < 0:
            self.top_p = None

        if self.top_k and self.top_k < 0:
            self.top_k = None

        if self.min_p and self.min_p < 0:
            self.min_p = None

        if self.temperature and self.temperature < 0:
            self.temperature = None

        if self.repetition_penalty and self.repetition_penalty < 0:
            self.repetition_penalty = None

        if self.presence_penalty and self.presence_penalty < 0:
            self.presence_penalty = None

        if self.frequency_penalty and self.frequency_penalty < 0:
            self.frequency_penalty = None

        if self.seed and self.seed < 0:
            self.seed = None

        if self.max_tokens is not None and self.max_tokens <= 0:
            self.max_tokens = None

        if not self.messages:
            raise ValueError("Messages cannot be empty")

        for message in self.messages:
            if not message.content:
                raise ValueError(f"Message content cannot be empty: {message}")
            if isinstance(message.content, list):
                for block in message.content:
                    if block is None:
                        raise ValueError(f"Block cannot be None: {message}")

        if self.messages[-1].role not in self._valid_last_message_roles:
            raise ValueError(
                f"Last message role must be one of {self._valid_last_message_roles}, but got {self.messages[-1].role}"
            )

        # Check tools and tool choice
        if self.tools and self.tool_choice and self.tool_choice.type == "tool":
            if not self.tools:
                raise ValueError("Tool choice is set, but no tools are provided.")
            if self.tool_choice.name not in [tool.name for tool in self.tools]:
                raise ValueError(
                    f"Tool choice '{self.tool_choice}' is not in the provided tools."
                )

        if not self.tools and self.tool_context:
            raise ValueError(
                "Tool context is provided, but no tools are specified. "
                "Please provide tools to use with the tool context."
            )

        # Apply global tool context to tools without specific context
        if self.tools is not None:
            global_tool_context = self.tool_context or []
            if global_tool_context:
                for tool in self.tools:
                    if tool.context is None:
                        tool.context = global_tool_context

        has_structured_output = bool(self.output_config and self.output_config.format)
        if self.response_format.type == ResponseFormatType.json_schema:
            has_structured_output = True

        # Check that we don't have tools when structured output is enabled
        if has_structured_output:
            if self.tools:
                if self.response_format.type == ResponseFormatType.json_schema:
                    raise ValueError(
                        "Tools are not supported when response_format is set to json_schema"
                    )
                raise ValueError(
                    "Tools are not supported when structured output is enabled."
                )
            if self.mcp_servers:
                raise ValueError(
                    "MCP servers are not supported when structured output is enabled."
                )
            if system.citations.enabled:
                raise ValueError(
                    "Citations are not supported when structured output is enabled."
                )

        # Check unique tools
        if self.tools:
            tool_names = [tool.name for tool in self.tools]
            if len(tool_names) != len(set(tool_names)):
                raise ValueError(
                    "Duplicate tool names found in the tools list."
                    f" Provided tools: {self.tools}"
                    f" Unique tool names: {set(tool_names)}"
                )

        # Check tool use and result blocks
        tool_uses_ids: set[str] = set()
        tool_results_ids: set[str] = set()
        for message in self.messages:
            if isinstance(message.content, list):
                for block in message.content:
                    if block is None:
                        raise ValueError("Block cannot be None")
                    elif isinstance(block, ToolUseBlock):
                        if message.role != "assistant":
                            raise ValueError(
                                f"Tool use blocks can only be used in assistant messages: {message}"
                            )
                        if block.id in tool_uses_ids:
                            raise ValueError(f"Duplicate tool use ID found: {block.id}")
                        tool_uses_ids.add(block.id)

                    elif isinstance(block, ToolResultBlock):
                        if block.tool_use_id not in tool_uses_ids:
                            raise ValueError(
                                f"Tool result block references an unknown tool use ID: {block.tool_use_id}"
                            )
                        tool_results_ids.add(block.tool_use_id)

                    elif isinstance(block, TLDRBlock):
                        if message.role != "assistant":
                            raise ValueError(
                                f"TLDR blocks can only be used in assistant messages: {message}"
                            )

        if tool_results_ids != tool_uses_ids:
            raise ValueError(
                "Tool result blocks must match the tool use IDs in the same message."
                f" Found tool use IDs: {tool_uses_ids}, but tool result IDs: {tool_results_ids}"
            )

        return self
