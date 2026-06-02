import re
from typing import Any, Literal

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.chat.extensions.citation import ZylonCitation
from private_gpt.chat.input_models import (
    ResponseFormatType,
    SystemExtensions,
)
from private_gpt.chat.schema_models import create_model_from_json_schema
from private_gpt.components.chat.models.chat_config_models import (
    CitationConfig,
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ResponseFormatConfig,
    SystemExtensionsConfig,
    ThinkingConfig,
    ToolSpec,
)
from private_gpt.components.tools.tool_pipeline import ToolPipeline
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.settings.settings import Settings


@singleton
class ChatRequestMapper:
    """Maps a ChatBody to a ChatRequest.

    This includes collecting tools from MCP servers and configuring the output schema.

    """

    @inject
    def __init__(
        self,
        settings: Settings,
        tool_pipeline: ToolPipeline,
    ) -> None:
        self._settings = settings
        self._tool_pipeline = tool_pipeline

    def _get_model(
        self,
        model_name: str | None,
    ) -> str | None:
        """Get the model name, defaulting if necessary."""
        if model_name:
            if model_name == "default":
                # To support Anthropic client, map "default" to the actual default model
                return None
            elif re.match(r"^claude-.*", model_name):
                # To support the client without setting the model,
                # map "claude-*" to the actual default model
                return None
        return model_name

    async def _tool_specs_to_internal(
        self,
        body: ChatBody,
    ) -> list[ToolSpec]:
        """Collect tools from the request."""
        if body.tool_choice.type == "none":
            return []

        output_tools: list[ToolSpec] = []
        if body.tools:
            for tool_spec in body.tools:
                output_tools.append(
                    ToolSpec(
                        type=tool_spec.type,
                        name=tool_spec.name,
                        description=tool_spec.description,
                        input_schema=tool_spec.input_schema,
                        context=tool_spec.context,
                        defer_loading=tool_spec.defer_loading,
                        instructions=tool_spec.instructions,
                    )
                )

        if body.tool_choice.type == "tool":
            output_tools = [
                filtered_tool
                for filtered_tool in output_tools
                if filtered_tool.name == body.tool_choice.name
            ]

        return output_tools

    async def _configure_output_cls_from_json_schema(
        self,
        request: ChatBody,
    ) -> type[BaseModel] | None:
        """Define the output schema based on the response format."""
        if (
            request.output_config
            and request.output_config.format
            and request.output_config.format.json_schema
        ):
            return create_model_from_json_schema(
                request.output_config.format.json_schema
            )

        if request.response_format.type == ResponseFormatType.json_schema:
            if not request.response_format.json_schema:
                raise ValueError(
                    "JSON schema must be provided when response format is json_schema"
                )
            return create_model_from_json_schema(request.response_format.json_schema)

        # Default to None for text responses
        return None

    async def _collect_sampling_params(
        self,
        request: ChatBody,
    ) -> dict[str, Any]:
        """Collect sampling parameters from the request request."""
        sampling_params: dict[str, Any] = {}
        if request.seed is not None:
            sampling_params["seed"] = request.seed
        if request.min_p is not None:
            sampling_params["min_p"] = request.min_p
        if request.top_p is not None:
            sampling_params["top_p"] = request.top_p
        if request.temperature is not None:
            sampling_params["temperature"] = request.temperature
        if request.top_k is not None:
            sampling_params["top_k"] = request.top_k
        if request.repetition_penalty is not None:
            sampling_params["repetition_penalty"] = request.repetition_penalty
        if request.presence_penalty is not None:
            sampling_params["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            sampling_params["frequency_penalty"] = request.frequency_penalty
        if request.max_tokens is not None:
            sampling_params["max_tokens"] = request.max_tokens

        return sampling_params

    async def get_thinking_config(
        self,
        request: ChatBody,
    ) -> ThinkingConfig:
        """Get the thinking config from the request."""
        thinking_enabled = (
            request.thinking.enabled
            or bool(request.output_config and request.output_config.effort)
            or bool(request.thinking.effort)
        )
        effort: Literal["low", "medium", "high", "max"] | None = (
            request.output_config.effort
            if request.output_config is not None
            and request.output_config.effort is not None
            else request.thinking.effort
        )
        if thinking_enabled:
            if effort is None:
                # Set default value when thinking is enabled
                effort = "medium"
        else:
            # Disable effort to avoid issues
            effort = None

        return ThinkingConfig(
            enabled=thinking_enabled,
            type=effort,
        )

    async def create_request_from_body(self, body: ChatBody) -> ResolvedChatRequest:
        """Create a ChatRequest from the ChatBody."""
        model_id = self._get_model(body.model)
        system = body.merged_system()
        tools = await self._tool_specs_to_internal(body)
        thinking = await self.get_thinking_config(body)
        output_cls = await self._configure_output_cls_from_json_schema(body)
        sampling_params = await self._collect_sampling_params(body)

        request = ResolvedChatRequest(
            stream=body.stream,
            messages=body.llama_index_messages(),
            system=ResolvedSystemConfig(
                model=model_id,
                prompt=system.text,
                use_default_prompt=system.use_default_prompt,
                correlation_id=body.correlation_id,
                priority=body.priority,
                extensions=SystemExtensionsConfig(
                    zylon_enabled=SystemExtensions.ZYLON in system.extensions,
                ),
                blob_visibility=system.blob_visibility,
                platform_prompts=system.prompt,
            ),
            tool_config=ResolvedToolConfig(
                tools=tools,
                tool_choices=(
                    body.tool_choice.name
                    if body.tool_choice.type == "tool" and body.tool_choice.name
                    else str(body.tool_choice.type)
                ),
                allow_parallel_tool_calls=not body.tool_choice.disable_parallel_tool_use,
                validation_mode=ToolValidationMode.from_str(
                    body.tool_choice.validation_mode
                ),
            ),
            tool_context=body.tool_context or [],
            context=ResolvedContextConfig(
                correlation_id=body.correlation_id,
                user_id=body.metadata.user_id if body.metadata else None,
                maximum_loaded_skills=(
                    body.maximum_loaded_skills
                    if body.maximum_loaded_skills is not None
                    else self._settings.skills.maximum_loaded_skills
                ),
            ),
            citation=CitationConfig(
                enabled=system.citations.enabled,
                citations=(
                    [
                        ZylonCitation.to_citation(citation)
                        for citation in system.citations.known_citations
                    ]
                    if system.citations.known_citations
                    else None
                ),
            ),
            thinking=thinking,
            response_format=ResponseFormatConfig(
                output_cls=output_cls,
            )
            if output_cls
            else None,
            sampling_params=sampling_params,
            mcp_servers=body.mcp_servers,
        )

        return request
