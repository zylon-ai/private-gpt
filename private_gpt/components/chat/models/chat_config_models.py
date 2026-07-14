import builtins
import enum
import inspect
import re
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Literal

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools.function_tool import AsyncCallable, _is_context_param
from llama_index.core.tools.utils import create_schema_from_function
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)

from private_gpt.chat.input_models import BlobVisibilityMode, PromptConfig
from private_gpt.chat.schema_models import create_model_from_json_schema
from private_gpt.components.engines.citations.types import Citation, Document
from private_gpt.components.llm.llm_helper import AsyncTokenizerFn, TokenizerFn
from private_gpt.components.sandbox.content_bundle import ContentBundle
from private_gpt.components.tools.tool_names import resolve_internal_tool_name
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.utils.artifact_input import ArtifactType
from private_gpt.settings.settings import LLMModelConfig


class LLMInstanceConfig(BaseModel):
    llm: LLM = Field(
        description="The LLM instance to use for the chat.",
    )
    config: LLMModelConfig = Field(
        description="The LLM model configuration.",
    )
    tokenizer: TokenizerFn | AsyncTokenizerFn | None = Field(
        default=None,
        description="The tokenizer function to use for the LLM.",
    )


class LLMConfig(BaseModel):
    main_model: LLMInstanceConfig = Field(
        description="The main LLM model to use for the chat.",
    )
    multimodal_image_model: LLMInstanceConfig | None = Field(
        default=None,
        description="The multimodal LLM model to use for image inputs in the chat.",
    )
    multimodal_audio_model: LLMInstanceConfig | None = Field(
        default=None,
        description="The multimodal LLM model to use for audio inputs in the chat.",
    )


class SystemExtensionsConfig(BaseModel):
    zylon_enabled: bool = Field(
        default=False,
        description="Whether Zylon extensions are enabled for the system prompt.",
    )


class SystemConfig(BaseModel):
    """Configuration for the system prompt."""

    model: str | None = Field(
        default=None, description="Model to use for the chat engine"
    )
    use_default_prompt: bool = Field(
        default=False,
        description=(
            "Deprecated: legacy toggle for built-in default prompt injection. "
            "Prefer explicit skills/layers for instruction composition."
        ),
        json_schema_extra={"deprecated": True},
    )
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for the request"
    )
    priority: int | None = Field(
        default=None,
        description="Priority of the request, used for scheduling",
    )
    extensions: SystemExtensionsConfig = Field(
        default_factory=SystemExtensionsConfig,
        description="Extensions configuration for the system prompt",
    )

    blob_visibility: BlobVisibilityMode = Field(
        default=BlobVisibilityMode.PUBLIC,
        description="Controls how blobs are exposed: binary (raw data), internal (private URI), or public (public URL)",
    )

    platform_prompts: PromptConfig = Field(
        default_factory=PromptConfig,
        description="Controls which platform-level prompt features are injected.",
    )

    def get_prompt(self) -> list[TextBlock] | None:
        """Get the system prompt, either from the model or the default."""
        return None


async def _dummy_tool_async_fn(**kwargs: Any) -> Any:
    """Default placeholder async function for tools without an implemented function."""
    raise RuntimeError(
        "Tool async_fn is not configured. Ensure internal tool contextualization "
        "or tool wiring runs before invocation."
    )


class ToolRequirements(enum.StrEnum):
    SANDBOX = "sandbox"


class ToolExecutionMetadata(BaseModel):
    rebuild_callable: str = Field(
        description="Import path to the callable that rebuilds the server tool."
    )
    rebuild_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable kwargs used to rebuild the server tool.",
    )

    MODEL_TAG_KEY: ClassVar[str] = "__pgpt_model__"

    @field_serializer("rebuild_kwargs", when_used="json")
    def _serialize_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Tag BaseModel values so they survive the JSON roundtrip."""
        result: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, BaseModel):
                result[key] = {
                    self.MODEL_TAG_KEY: f"{type(value).__module__}:{type(value).__qualname__}",
                    "data": value.model_dump(mode="json"),
                }
            else:
                result[key] = value
        return result

    @field_validator("rebuild_kwargs", mode="before")
    @classmethod
    def _deserialize_kwargs(cls, kwargs: Any) -> Any:
        """Untag BaseModel values after JSON roundtrip."""
        if not isinstance(kwargs, dict):
            return kwargs
        import importlib

        result: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, dict) and cls.MODEL_TAG_KEY in value:
                module_path, qualname = value[cls.MODEL_TAG_KEY].rsplit(":", 1)
                module = importlib.import_module(module_path)
                model_cls: Any = module
                for attribute in qualname.split("."):
                    model_cls = getattr(model_cls, attribute)
                result[key] = model_cls.model_validate(value["data"])
            else:
                result[key] = value
        return result


class ToolSpec(BaseModel):
    name: str | None = Field(description="Unique name identifier for the tool")
    type: str | None = Field(
        default=None,
        description="Type of the tool, use to identify internal tools database_query_v1 or semantic_search_v1",
    )
    runtime: Literal["client", "server"] = Field(
        default="client",
        description="Execution runtime for the tool. 'server' means "
        "the tool is executed by the server; 'client' means the call is passed back to the caller.",
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
            "When true, hide this tool from the LLM until at least one skill is "
            "loaded in the conversation."
        ),
    )
    async_fn: AsyncCallable = Field(
        default=_dummy_tool_async_fn,
        description="Asynchronous function implementation of the tool",
    )
    async_callback: AsyncCallable | None = Field(
        default=None,
        description="Asynchronous callback function for the tool",
    )
    partial_params: dict[str, Any] | None = Field(
        default=None,
        description="Predefined parameters to be used when invoking the tool",
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "Optional instructions injected into the system prompt when this tool "
            "is available. For internal tools a default template is used; providing "
            "a value here overrides that default. Set to an empty string to disable."
        ),
    )
    requirements: list[ToolRequirements] = Field(
        default_factory=list,
        description="List of requirements for the tool, e.g., SANDBOX",
    )
    execution_metadata: ToolExecutionMetadata | None = Field(
        default=None,
        description=(
            "Optional metadata used to rebuild this server tool in another process."
        ),
    )

    @field_serializer("async_fn", "async_callback", when_used="json")
    def _serialize_callable(self, _v: Any) -> None:
        return None

    @field_validator("async_fn", mode="before")
    @classmethod
    def _deserialize_callable(cls, v: Any) -> Any:
        """Restore callable after JSON deserialization."""
        return _dummy_tool_async_fn if v is None else v

    def get_original_tool_name(self) -> str:
        """Get the original tool name without version suffix."""
        potential_tool_name: str = self.type or self.name or ""
        if not potential_tool_name:
            raise ValueError("Tool must have at least a name or a type.")
        resolved_internal_name = resolve_internal_tool_name(potential_tool_name)
        if resolved_internal_name is not None:
            return resolved_internal_name
        return re.sub(r"_v\d+$", "", potential_tool_name)

    @classmethod
    def from_defaults(
        cls,
        name: str,
        type: str | None = None,
        runtime: Literal["client", "server"] = "client",
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        context: list[ArtifactType] | None = None,
        defer_loading: bool = False,
        async_fn: AsyncCallable | None = None,
        async_callback: AsyncCallable | None = None,
        partial_params: dict[str, Any] | None = None,
        instructions: str | None = None,
        requirements: list[ToolRequirements] | None = None,
        execution_metadata: ToolExecutionMetadata | None = None,
    ) -> "ToolSpec":
        """Create a ToolSpec from default parameters."""
        if not input_schema and not async_fn:
            raise ValueError(
                "At least an input schema, async function, or async callback must be provided."
            )

        if not input_schema and async_fn is not None:
            schema = cls.build_fn_schema(async_fn, partial_params)
            input_schema = schema.model_json_schema()

        return cls(
            name=name,
            type=type,
            runtime=runtime,
            description=description,
            input_schema=input_schema,
            context=context,
            defer_loading=defer_loading,
            async_fn=async_fn or _dummy_tool_async_fn,
            async_callback=async_callback,
            partial_params=partial_params,
            instructions=instructions,
            requirements=requirements or [],
            execution_metadata=execution_metadata,
        )

    @classmethod
    def from_llama_index(
        cls,
        tool: BaseTool | Callable[..., Any],
    ) -> "ToolSpec":
        """Create ToolSpec from LlamaIndex FunctionTool."""
        if not isinstance(tool, BaseTool):
            raise ValueError("Unsupported tool type. Expected a FunctionTool.")

        schema: dict[str, Any] = {}
        if tool.metadata.fn_schema:
            schema = tool.metadata.fn_schema.model_json_schema()

        return ToolSpec(
            # TODO: re-check when we had removed return_direct
            type=tool.metadata.name if not tool.metadata.return_direct else None,
            runtime="server" if not tool.metadata.return_direct else "client",
            name=tool.metadata.name,
            description=tool.metadata.description,
            input_schema=schema,
            defer_loading=False,
            async_fn=tool.async_fn
            if hasattr(tool, "async_fn")
            else _dummy_tool_async_fn,
            async_callback=tool._async_callback
            if hasattr(tool, "_async_callback")
            else None,
            partial_params=tool.partial_params
            if hasattr(tool, "partial_params")
            else None,
            execution_metadata=None,
        )

    def to_function_tool(self) -> "FunctionTool":
        """Convert into LlamaIndex tool."""
        schema = self.input_schema or {"type": "object", "properties": {}}
        model_schema = create_model_from_json_schema(
            schema, model_name=f"{self.name}_schema"
        )

        return FunctionTool.from_defaults(
            name=self.name,
            description=self.description,
            # This is still a llama-index tool,
            # the logic is inverted, return_direct=True => the tool is not executed,
            # just return the function call
            # So for server tools, we want to execute them directly,
            # For the rest (user provided), we pass the turn back to the caller
            return_direct=self.runtime != "server",
            fn_schema=model_schema if model_schema else None,
            async_fn=self.async_fn,
            async_callback=self.async_callback,
            partial_params=self.partial_params,
        )

    @staticmethod
    def build_fn_schema(
        fn: Callable[..., Any] | Callable[..., Awaitable[Any]],
        partial_params: dict[str, Any] | None = None,
    ) -> builtins.type[BaseModel]:
        partial_params = partial_params or {}
        sig = inspect.signature(fn)
        fn_params = set(sig.parameters.keys())

        docstring = fn.__doc__ or ""
        param_docs, _ = FunctionTool.extract_param_docs(docstring, fn_params)

        ignore_fields: list[str] = []

        for param in sig.parameters.values():
            if _is_context_param(param.annotation):
                ignore_fields.append(param.name)
            elif param.name == "self":
                ignore_fields.append("self")

        ignore_fields.extend(partial_params.keys())

        fn_schema = create_schema_from_function(
            fn.__name__,
            fn,
            additional_fields=None,
            ignore_fields=ignore_fields,
        )

        if fn_schema is not None and param_docs:
            for param_name, field in fn_schema.model_fields.items():
                if not field.description and param_name in param_docs:
                    field.description = param_docs[param_name].strip()

        return fn_schema


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    tool_choices: str | list[str] = Field(
        default="auto",
        description="The tool choice for the agent. "
        "Must be 'auto' or the name of a tools.",
    )
    allow_parallel_tool_calls: bool = Field(
        default=True,
        description="Whether to allow parallel tool calls.",
    )
    validation_mode: ToolValidationMode = Field(
        default=ToolValidationMode.LAZY,
        description="The tool validation mode. Can be 'eager' or 'lazy'.",
    )


class ContextConfig(BaseModel):
    add_context_to_system_prompt: bool = Field(
        default=False,
        description="Whether to add context to the system prompt.",
    )
    deduplicate_context_in_history: bool = Field(
        default=False,
        description="Whether to deduplicate nodes in the chat history to avoid sending "
        "the same document multiple times to the LLM. If enabled, "
        "it will keep the last occurrence of the document in the chat history.",
    )
    maximum_context_length: int | None = Field(
        default=None,
        description="Maximum length of context to use for the chat.",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for the chat session.",
    )
    user_id: str | None = Field(
        default=None,
        description="Opaque user identifier for the chat session.",
    )
    container: str | None = Field(
        default=None,
        description="Container identifier for reuse across requests.",
    )
    maximum_loaded_skills: int | None = Field(
        default=None,
        description=(
            "Maximum number of concurrently loaded skills allowed in the chat."
        ),
        ge=1,
    )


class CitationConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Whether to enable citations in the chat.",
    )
    citations: list[Citation] | None = Field(
        default=None,
        description="List of citations to use in the chat.",
    )
    force_to_return_citations: bool = Field(
        default=False,
        description="Whether to force the LLM to return citations.",
    )
    return_missing_citations: bool = Field(
        default=False,
        description="Whether to return all missing citations.",
    )


class CondensationConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Whether to enable condensation in the chat.",
    )


class ThinkingConfig(BaseModel):

    enabled: bool = Field(
        default=False,
        description="Whether to enable reasoning in the chat.",
    )
    type: Literal["low", "medium", "high", "max", "xhigh"] | None = Field(
        default="medium",
        description="The level of reasoning to use in the chat.",
    )


class ResponseFormatConfig(BaseModel):
    """Configuration for the response format."""

    output_cls: type[BaseModel] | None = Field(
        default=None, description="Output class to use for the response format"
    )

    @field_serializer("output_cls", when_used="json")
    def _serialize_output_cls(
        self,
        output_cls: type[BaseModel] | None,
    ) -> dict[str, Any] | None:
        return output_cls.model_json_schema() if output_cls is not None else None

    @field_validator("output_cls", mode="before")
    @classmethod
    def _deserialize_output_cls(cls, output_cls: Any) -> Any:
        if isinstance(output_cls, dict):
            return create_model_from_json_schema(output_cls)
        return output_cls


class ChatRequest(BaseModel):
    """Request model for chat-based engines with agent capabilities.

    Treat this type as immutable after creation to avoid common bugs,
    always create a new instance using `model_copy` if you need to modify it.
    """

    stream: bool = Field(
        default=False,
        description="Whether to stream the response or return it all at once.",
    )
    messages: list[ChatMessage] = Field(
        description="List of chat messages in the conversation."
    )
    system: SystemConfig = Field(
        default_factory=SystemConfig,
        description="Configuration for the system prompt.",
    )
    tool_config: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Configuration for tools.",
    )
    tool_context: list[ArtifactType] = Field(
        default_factory=list,
        description="Context for internal tools",
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Configuration for context handling.",
    )
    condensation: CondensationConfig = Field(
        default_factory=CondensationConfig,
        description="Configuration for condensation handling.",
    )
    citation: CitationConfig = Field(
        default_factory=CitationConfig,
        description="Configuration for citation handling.",
    )
    thinking: ThinkingConfig = Field(
        default_factory=ThinkingConfig,
        description="Configuration for reasoning/thinking handling.",
    )
    response_format: ResponseFormatConfig | None = Field(
        default=None,
        description="Configuration for response formatting.",
    )
    sampling_params: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for sampling in the LLM."
    )
    mcp_servers: list[McpServerConfig] = Field(
        default_factory=list,
        description="List of MCP server configurations. "
        "Tools are fetched at runtime before inference time"
        "and added to the tools list.",
    )

    def to_messages(self) -> list[ChatMessage]:
        """Convert the ChatRequest into a list of ChatMessages for LLM input."""
        final_messages = [
            message for message in self.messages if message.role != MessageRole.SYSTEM
        ]

        prompt_blocks = self.system.get_prompt()
        if prompt_blocks:
            system_message = ChatMessage(role=MessageRole.SYSTEM, blocks=prompt_blocks)
            final_messages = [system_message, *final_messages]

        return final_messages


class ResolvedSystemConfig(SystemConfig):
    """Consolidated version of SystemConfig."""

    prompt: str | list[TextBlock] | None = Field(
        default=None,
        description="The system prompt to use for the chat.",
    )

    def get_prompt(self) -> list[TextBlock] | None:
        prompt_block = (
            [TextBlock(text=self.prompt)]
            if isinstance(self.prompt, str)
            else self.prompt
        )
        return prompt_block or None


class ResolvedToolConfig(ToolConfig):
    """Consolidated version of ToolConfig."""

    tools: list[ToolSpec] = Field(
        default_factory=list,
        description="Tools to use for the chat.",
    )


class ResolvedContextConfig(ContextConfig):
    """Consolidated version of ContextConfig."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document] | None = Field(
        default=None,
        description="List of documents to use as context in the chat.",
    )
    content_bundles: list[ContentBundle] = Field(
        default_factory=list,
        description=(
            "Content bundles transferred from ContentBundlesLayer. "
            "Consumed by tool builders (e.g. BashToolBuilder) to mount skills."
        ),
        exclude=True,
    )


class ResolvedChatRequest(ChatRequest):
    """Consolidated version of ChatRequest with flattened fields for easier access."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system: ResolvedSystemConfig = Field(
        default_factory=ResolvedSystemConfig,
        description="Configuration for the system prompt.",
    )
    tool_config: ResolvedToolConfig = Field(
        default_factory=ResolvedToolConfig,
        description="Configuration for tools.",
    )
    context: ResolvedContextConfig = Field(
        default_factory=ResolvedContextConfig,
        description="Configuration for context handling.",
    )
