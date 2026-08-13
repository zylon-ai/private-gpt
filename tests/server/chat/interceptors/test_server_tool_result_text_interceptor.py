from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.events.models import NO_TOOL_CONTENT, ServerToolResultBlock
from private_gpt.server.chat.interceptors.server_tool_result_text_interceptor import (
    ServerToolResultTextInterceptor,
)


@pytest.mark.anyio
async def test_empty_rendered_server_result_gets_placeholder() -> None:
    message = ChatMessage(
        role=MessageRole.TOOL,
        content="stale",
        additional_kwargs={
            "server_tool_result": [
                ServerToolResultBlock(tool_use_id="tc_1", content="")
            ],
        },
    )
    state = SimpleNamespace(
        input=SimpleNamespace(request=SimpleNamespace(messages=[message]))
    )
    context = SimpleNamespace(phase=InterceptorPhase.BEFORE_ITERATION, state=state)
    context.set_state = lambda new_state: None

    await ServerToolResultTextInterceptor().intercept(context)

    assert message.content == NO_TOOL_CONTENT
    assert message.blocks[0].text == NO_TOOL_CONTENT


@pytest.mark.anyio
async def test_server_result_rendering_is_idempotent() -> None:
    message = ChatMessage(
        role=MessageRole.TOOL,
        additional_kwargs={
            "server_tool_result": [
                ServerToolResultBlock(tool_use_id="tc_1", content="ok")
            ],
        },
    )
    state = SimpleNamespace(
        input=SimpleNamespace(request=SimpleNamespace(messages=[message]))
    )
    context = SimpleNamespace(phase=InterceptorPhase.BEFORE_ITERATION, state=state)
    context.set_state = lambda new_state: None
    interceptor = ServerToolResultTextInterceptor()

    await interceptor.intercept(context)
    first = message.content
    await interceptor.intercept(context)

    assert first == "ok"
    assert message.content == "ok"
