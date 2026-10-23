from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.tools.tool_names import SKILL_LOAD_TOOL_NAME
from private_gpt.events.models import (
    NO_TOOL_CONTENT,
    BashCodeExecutionResultBlock,
    ServerToolResultBlock,
)
from private_gpt.server.chat.interceptors.server_tool_result_text_interceptor import (
    ServerToolResultTextInterceptor,
)
from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
    _resolve_active_skill_names,
)


def _context(message: ChatMessage) -> SimpleNamespace:
    state = SimpleNamespace(
        input=SimpleNamespace(request=SimpleNamespace(messages=[message]))
    )
    context = SimpleNamespace(phase=InterceptorPhase.BEFORE_ITERATION, state=state)
    context.set_state = lambda new_state: None
    return context


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
    context = _context(message)

    await ServerToolResultTextInterceptor().intercept(context)
    rendered = context.state.input.request.messages[0]

    assert rendered is not message
    assert message.content == "stale"
    assert rendered.content == NO_TOOL_CONTENT
    assert rendered.blocks[0].text == NO_TOOL_CONTENT
    assert "server_tool_result" in rendered.additional_kwargs


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
    context = _context(message)
    interceptor = ServerToolResultTextInterceptor()

    await interceptor.intercept(context)
    first = context.state.input.request.messages[0]
    await interceptor.intercept(context)
    second = context.state.input.request.messages[0]

    assert first.content == "ok"
    assert second.content == "ok"
    assert second is first


@pytest.mark.anyio
async def test_bash_result_in_additional_kwargs_is_rendered() -> None:
    result = BashCodeExecutionResultBlock(
        stdout="# Potato\n",
        stderr="",
        return_code=0,
    )
    message = ChatMessage(
        role=MessageRole.TOOL,
        content="(no-output)",
        additional_kwargs={
            "tool_call_name": "bash",
            "tool_call_args": {"command": "ls"},
            "bash_code_execution_result": [result],
        },
    )
    context = _context(message)

    await ServerToolResultTextInterceptor().intercept(context)
    rendered = context.state.input.request.messages[0]

    assert rendered is not message
    assert rendered.content == result.render()
    assert "# Potato" in rendered.content
    assert rendered.additional_kwargs["bash_code_execution_result"] is not None
    assert "bash_code_execution_result" in message.additional_kwargs
    assert rendered.additional_kwargs["tool_call_name"] == "bash"
    assert rendered.additional_kwargs["tool_call_args"] == {"command": "ls"}


@pytest.mark.anyio
async def test_does_not_mutate_original_tool_message_kwargs() -> None:
    original_kwargs = {
        "tool_call_id": "tc_1",
        "tool_call_name": "bash",
        "tool_call_args": {"command": "ls"},
        "server_tool_result": [ServerToolResultBlock(tool_use_id="tc_1", content="ok")],
    }
    message = ChatMessage(
        role=MessageRole.TOOL,
        content="stale",
        additional_kwargs=original_kwargs,
    )
    message_kwargs = message.additional_kwargs
    context = _context(message)

    await ServerToolResultTextInterceptor().intercept(context)
    rendered = context.state.input.request.messages[0]

    assert rendered is not message
    assert message.additional_kwargs is message_kwargs
    assert "server_tool_result" in message_kwargs
    assert message.content == "stale"
    assert rendered.content == "ok"
    assert "server_tool_result" in rendered.additional_kwargs
    assert rendered.additional_kwargs is not message_kwargs
    assert rendered.additional_kwargs["tool_call_name"] == "bash"
    assert rendered.additional_kwargs["tool_call_args"] == {"command": "ls"}


@pytest.mark.anyio
async def test_load_skill_keeps_kwargs_so_skill_state_can_be_reconstructed() -> None:
    payload = '{"name": "skill-creator", "loaded": true}'
    message = ChatMessage(
        role=MessageRole.TOOL,
        content=payload,
        additional_kwargs={
            "tool_call_id": "tc_1",
            "tool_call_name": SKILL_LOAD_TOOL_NAME,
            "tool_call_args": {"name": "skill-creator"},
            "server_tool_result": [
                ServerToolResultBlock(tool_use_id="tc_1", content="")
            ],
        },
    )
    context = _context(message)

    await ServerToolResultTextInterceptor().intercept(context)
    rendered = context.state.input.request.messages[0]

    assert message.content == payload
    assert "server_tool_result" in rendered.additional_kwargs
    assert rendered.additional_kwargs["tool_call_name"] == SKILL_LOAD_TOOL_NAME
    assert rendered.additional_kwargs["tool_call_args"] == {"name": "skill-creator"}
    assert _resolve_active_skill_names([rendered]) == {"skill-creator"}
