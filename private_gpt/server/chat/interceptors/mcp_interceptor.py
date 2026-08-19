import asyncio
from collections.abc import Callable
from typing import Any, Literal

import httpx
import httpx2
from injector import inject, singleton
from pydantic import BaseModel, Field, ValidationError

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionInterceptorContext,
)
from private_gpt.events.event_errors import Errors
from private_gpt.events.models import (
    Event,
    McpTokensRefreshedEvent,
    McpTokensRefreshFailedEvent,
    RawContentBlockStartEvent,
)
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import (
    MCP_PREVIOUS_REFRESH_TOKEN_KEY,
    MCP_REFRESH_FAILED_KEY,
    MCP_TOKEN_REFRESH_KEY,
    McpService,
    mcp_tool_to_spec,
)


class _McpTokenRefreshPayload(BaseModel):
    status: Literal["success", "failure"]
    name: str
    url: str
    previous_refresh_token: str | None = None
    authorization_token: str | None = None
    refresh_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _extract_original_exception(exc: BaseException) -> BaseException:
    if isinstance(exc, (httpx.HTTPStatusError, httpx2.HTTPStatusError)):
        if exc.response.status_code in (401, 403):
            return PermissionError(
                f"MCP server rejected the request with HTTP {exc.response.status_code}. "
                "Check your authorization token."
            )
        return exc

    if isinstance(exc, BaseExceptionGroup):
        if len(exc.exceptions) == 1:
            return _extract_original_exception(exc.exceptions[0])
        for sub in exc.exceptions:
            found = _extract_original_exception(sub)
            if found is not exc:
                return found

    return exc


def _pop_token_refresh_payload(
    config: McpServerConfig,
) -> _McpTokenRefreshPayload | None:
    previous_refresh_token = config.metadata.pop(MCP_PREVIOUS_REFRESH_TOKEN_KEY, None)
    refresh_failed = config.metadata.pop(MCP_REFRESH_FAILED_KEY, False)
    if (
        isinstance(previous_refresh_token, str)
        and config.authorization_token
        and config.refresh_token
    ):
        return _McpTokenRefreshPayload(
            status="success",
            name=config.name or "mcp",
            url=config.url,
            previous_refresh_token=previous_refresh_token,
            authorization_token=config.authorization_token,
            refresh_token=config.refresh_token,
            metadata=dict(config.metadata),
        )
    if refresh_failed:
        return _McpTokenRefreshPayload(
            status="failure",
            name=config.name or "mcp",
            url=config.url,
            previous_refresh_token=config.refresh_token,
            metadata=dict(config.metadata),
        )
    return None


@singleton
class McpRequestInterceptor(
    ChatRequestLoopInterceptor,
    ChatResponseLoopInterceptor,
    ToolExecutionInterceptor,
):
    @inject
    def __init__(self, mcp_service: McpService) -> None:
        self._mcp_service = mcp_service

    @staticmethod
    def _event_from_payload(
        payload: _McpTokenRefreshPayload,
    ) -> McpTokensRefreshedEvent | McpTokensRefreshFailedEvent | None:
        if payload.status == "success":
            if not payload.authorization_token or not payload.refresh_token:
                return None
            return McpTokensRefreshedEvent(
                name=payload.name,
                url=payload.url,
                authorization_token=payload.authorization_token,
                refresh_token=payload.refresh_token,
                metadata=payload.metadata,
            )
        return McpTokensRefreshFailedEvent(
            name=payload.name,
            url=payload.url,
            error="MCP OAuth token refresh failed",
            metadata=payload.metadata,
        )

    @staticmethod
    def _configs(input_state: Any) -> list[McpServerConfig]:
        configs = list(input_state.request.mcp_servers)
        for layer in input_state.context_stack.layers:
            if not isinstance(layer, ToolDefinitionsLayer) or layer.source != "mcp":
                continue
            for tool in layer.tools:
                metadata = tool.execution_metadata
                config = metadata and metadata.rebuild_kwargs.get("config")
                if isinstance(config, McpServerConfig):
                    configs.append(config)
        return configs

    @classmethod
    def _apply_token_refresh(
        cls, context: ChatInterceptorContext, payload: _McpTokenRefreshPayload
    ) -> bool:
        configs = [
            config
            for input_state in (context.state.input, context.state.original_input)
            if input_state is not None
            for config in cls._configs(input_state)
            if (config.name or "mcp") == payload.name and config.url == payload.url
        ]
        if payload.status == "success":
            if not payload.previous_refresh_token:
                return False
            updated = False
            for config in configs:
                if config.refresh_token != payload.previous_refresh_token:
                    continue
                if not payload.authorization_token or not payload.refresh_token:
                    continue
                config.authorization_token = payload.authorization_token
                config.refresh_token = payload.refresh_token
                config.metadata.update(payload.metadata)
                config.metadata.pop(MCP_PREVIOUS_REFRESH_TOKEN_KEY, None)
                config.metadata.pop(MCP_REFRESH_FAILED_KEY, None)
                updated = True
            return updated

        if payload.previous_refresh_token is None:
            return bool(configs)
        return any(
            config.refresh_token == payload.previous_refresh_token for config in configs
        )

    @staticmethod
    def _pop_message_payloads(
        context: ChatInterceptorContext,
    ) -> list[_McpTokenRefreshPayload]:
        payloads: list[_McpTokenRefreshPayload] = []
        for input_state in (context.state.input, context.state.original_input):
            if input_state is None:
                continue
            for message in input_state.request.messages:
                raw_payload = message.additional_kwargs.pop(MCP_TOKEN_REFRESH_KEY, None)
                if raw_payload is None:
                    continue
                try:
                    payloads.append(_McpTokenRefreshPayload.model_validate(raw_payload))
                except ValidationError:
                    continue
        return payloads

    @classmethod
    def _consume_token_refreshes(cls, context: ChatInterceptorContext) -> None:
        consumed_servers: set[tuple[str, str]] = set()
        for payload in cls._pop_message_payloads(context):
            identity = (payload.name, payload.url)
            if identity in consumed_servers:
                continue
            if not cls._apply_token_refresh(context, payload):
                continue
            event = cls._event_from_payload(payload)
            if event is None:
                continue
            consumed_servers.add(identity)
            context.emit_event(event)

    async def _collect_tools_from_mcp(
        self,
        request: ChatRequest,
        raise_on_error: bool = True,
        emit_event: Callable[[Event], None] | None = None,
    ) -> list[ToolSpec]:
        try:
            output_tools: list[ToolSpec] = []
            if request.mcp_servers:

                async def _get_mcp_tools(
                    config: McpServerConfig,
                ) -> list[ToolSpec]:
                    """Fetch tools from a single MCP server."""
                    client = self._mcp_service.create_client(config)
                    try:
                        try:
                            mcp_tools = await client.list_tools()
                        finally:
                            payload = _pop_token_refresh_payload(config)
                            event = (
                                self._event_from_payload(payload)
                                if payload is not None
                                else None
                            )
                            if emit_event and event is not None:
                                emit_event(event)
                        return [mcp_tool_to_spec(config, tool) for tool in mcp_tools]
                    finally:
                        await client.close()

                # Gather tools from all MCP servers concurrently
                mcp_tools_results: list[BaseException | list[ToolSpec]] = list(
                    await asyncio.gather(
                        *(
                            _get_mcp_tools(mcp_server_config)
                            for mcp_server_config in request.mcp_servers
                        ),
                        return_exceptions=True,
                    )
                )

                # Find any issue with the connection
                for result_or_exception in mcp_tools_results:
                    if isinstance(result_or_exception, BaseException):
                        exception = result_or_exception
                        raise _extract_original_exception(exception)

                # flatten the list of tools
                mcp_tools: list[ToolSpec] = [
                    mcp_tool
                    for tools in mcp_tools_results
                    if isinstance(tools, list)
                    for mcp_tool in tools
                ]
                output_tools.extend(mcp_tools)

            return output_tools

        except Exception as e:
            if raise_on_error:
                raise e
            return []

    async def intercept(
        self, context: ChatInterceptorContext | ToolExecutionInterceptorContext
    ) -> None:
        if isinstance(context, ToolExecutionInterceptorContext):
            if context.phase != InterceptorPhase.AFTER_TOOL or context.response is None:
                return
            metadata = context.request.tool_spec.execution_metadata
            config = metadata and metadata.rebuild_kwargs.get("config")
            if isinstance(config, McpServerConfig):
                payload = _pop_token_refresh_payload(config)
                if payload is not None:
                    context.response.tool_message.additional_kwargs[
                        MCP_TOKEN_REFRESH_KEY
                    ] = payload.model_dump(mode="json", exclude_none=True)
            return
        if context.phase == InterceptorPhase.AFTER_ITERATION:
            self._consume_token_refreshes(context)
            return
        if (
            context.phase != InterceptorPhase.VALIDATION
            and context.phase != InterceptorPhase.BEFORE_ITERATION
        ):
            return

        try:
            state = context.state
            request = state.input.request

            mcp_tools = await self._collect_tools_from_mcp(
                request,
                emit_event=context.emit_event,
            )
            if mcp_tools:
                state.input.context_stack = (
                    state.input.context_stack.remove_layers_of_source("mcp")
                )
                state.input.context_stack = state.input.context_stack.append_layer(
                    ToolDefinitionsLayer(tools=mcp_tools, source="mcp")
                )
                state.input.request.mcp_servers = []

                # Update original context stack as well
                if state.original_input:
                    state.original_input.context_stack = (
                        state.original_input.context_stack.remove_layers_of_source(
                            "mcp"
                        ).append_layer(
                            ToolDefinitionsLayer(tools=mcp_tools, source="mcp")
                        )
                    )
                    state.original_input.request.mcp_servers = []

            context.set_state(state)
        except (ImportError, ModuleNotFoundError):
            raise
        except PermissionError as e:
            # Wrap the error to give details about what it is the real error
            raise Errors.build(e, Errors.Codes.PERMISSION_MCP_AUTH_ERROR) from e
        except Exception as e:
            raise Errors.InvalidRequest(
                "Failed to fetch tools from MCP servers.",
                event_code=Errors.Codes.INVALID_REQUEST_INVALID_MCP_ERROR,
            ) from e

    async def intercept_event(
        self,
        event: Event,
        context: ChatInterceptorContext,
    ) -> Event:
        if isinstance(event, RawContentBlockStartEvent):
            self._consume_token_refreshes(context)
        return event
