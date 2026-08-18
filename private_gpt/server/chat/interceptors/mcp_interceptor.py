import asyncio
from collections.abc import Callable

import httpx
import httpx2
from injector import inject, singleton

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
    McpService,
    mcp_tool_to_spec,
)


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


def _pop_token_refresh_event(
    config: McpServerConfig,
) -> McpTokensRefreshedEvent | McpTokensRefreshFailedEvent | None:
    previous_refresh_token = config.metadata.pop(MCP_PREVIOUS_REFRESH_TOKEN_KEY, None)
    refresh_failed = config.metadata.pop(MCP_REFRESH_FAILED_KEY, False)
    if (
        isinstance(previous_refresh_token, str)
        and config.authorization_token
        and config.refresh_token
    ):
        return McpTokensRefreshedEvent(
            name=config.name or "mcp",
            url=config.url,
            previous_refresh_token=previous_refresh_token,
            authorization_token=config.authorization_token,
            refresh_token=config.refresh_token,
            metadata=config.metadata,
        )
    if refresh_failed:
        return McpTokensRefreshFailedEvent(
            name=config.name or "mcp",
            url=config.url,
            error="MCP OAuth token refresh failed",
            metadata=config.metadata,
        )
    return None


@singleton
class McpRequestInterceptor(ChatRequestLoopInterceptor, ChatResponseLoopInterceptor):
    @inject
    def __init__(self, mcp_service: McpService) -> None:
        self._mcp_service = mcp_service

    @staticmethod
    def _consume_token_refreshes(context: ChatInterceptorContext) -> None:
        for tool in context.state.input.context_stack.all_tools():
            metadata = tool.execution_metadata
            config = metadata and metadata.rebuild_kwargs.get("config")
            if not isinstance(config, McpServerConfig):
                continue
            event = _pop_token_refresh_event(config)
            if event is None:
                continue
            for input_state in (context.state.input, context.state.original_input):
                if input_state is None:
                    continue
                for state_tool in input_state.context_stack.all_tools():
                    state_metadata = state_tool.execution_metadata
                    state_config = state_metadata and state_metadata.rebuild_kwargs.get(
                        "config"
                    )
                    if (
                        isinstance(state_config, McpServerConfig)
                        and (state_config.name or "mcp") == event.name
                        and state_config.url == event.url
                    ):
                        if isinstance(
                            event, McpTokensRefreshedEvent
                        ) and state_config.refresh_token in (
                            event.previous_refresh_token,
                            event.refresh_token,
                        ):
                            state_config.authorization_token = event.authorization_token
                            state_config.refresh_token = event.refresh_token
                        state_config.metadata.pop(MCP_PREVIOUS_REFRESH_TOKEN_KEY, None)
                        state_config.metadata.pop(MCP_REFRESH_FAILED_KEY, None)
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
                            event = _pop_token_refresh_event(config)
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

    async def intercept(self, context: ChatInterceptorContext) -> None:
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
