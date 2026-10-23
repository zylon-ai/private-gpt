import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.context.models.context_layer import RuntimeInstructionsLayer
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.server.chat.interceptors.loop_detection_interceptor import (
    LoopDetectionRequestInterceptor,
    LoopDetectionResult,
)
from private_gpt.settings.settings import ChatSettings


def _interceptor(
    interval: int | None,
) -> tuple[LoopDetectionRequestInterceptor, MagicMock]:
    settings = MagicMock()
    settings.chat.loop_detection_interval = interval
    builder = MagicMock()
    builder.create_loop_detection_prompt.return_value = MagicMock()
    return LoopDetectionRequestInterceptor(settings, builder), builder


def _context(messages: list[ChatMessage], *, is_loop: bool = False) -> MagicMock:
    context = MagicMock()
    context.phase = InterceptorPhase.BEFORE_ITERATION
    context.state.input.request.to_messages.return_value = messages
    context.state.input.context_stack = ContextStack()
    context.llm.astructured_predict = AsyncMock(
        return_value=LoopDetectionResult(is_loop=is_loop, reason="test")
    )
    return context


@pytest.mark.parametrize("value", [None, 0, -3])
def test_non_positive_interval_disables_loop_detection(value: int | None) -> None:
    assert ChatSettings(loop_detection_interval=value).loop_detection_interval is None


@pytest.mark.asyncio
async def test_evaluates_each_n_assistant_messages_after_latest_user() -> None:
    interceptor, builder = _interceptor(2)
    context = _context(
        [
            ChatMessage(role=MessageRole.ASSISTANT, content="old"),
            ChatMessage(role=MessageRole.USER, content="new request"),
            ChatMessage(role=MessageRole.ASSISTANT, content="call one"),
            ChatMessage(role=MessageRole.TOOL, content="result one"),
            ChatMessage(role=MessageRole.ASSISTANT, content="call two"),
        ]
    )

    await interceptor.intercept(context)

    builder.create_loop_detection_prompt.assert_called_once()
    conversation = builder.create_loop_detection_prompt.call_args.kwargs["conversation"]
    examples = builder.create_loop_detection_prompt.call_args.kwargs["examples"]
    assert "old" not in conversation
    assert "new request" in conversation
    parsed_examples = json.loads(examples)
    assert {example["result"]["is_loop"] for example in parsed_examples} == {
        True,
        False,
    }
    assert all(example["result"]["reason"] for example in parsed_examples)
    context.llm.astructured_predict.assert_awaited_once_with(
        output_cls=LoopDetectionResult,
        prompt=builder.create_loop_detection_prompt.return_value,
        llm_kwargs={"max_tokens": 128},
    )


@pytest.mark.asyncio
async def test_does_not_evaluate_between_intervals() -> None:
    interceptor, builder = _interceptor(2)
    context = _context(
        [
            ChatMessage(role=MessageRole.USER, content="request"),
            ChatMessage(role=MessageRole.ASSISTANT, content="first"),
        ]
    )

    await interceptor.intercept(context)

    builder.create_loop_detection_prompt.assert_not_called()
    context.llm.astructured_predict.assert_not_awaited()


@pytest.mark.asyncio
async def test_detected_loop_replaces_entire_context_stack() -> None:
    interceptor, _ = _interceptor(1)
    context = _context(
        [
            ChatMessage(role=MessageRole.USER, content="request"),
            ChatMessage(role=MessageRole.ASSISTANT, content="same action"),
        ],
        is_loop=True,
    )
    context.state.input.context_stack = MagicMock()

    await interceptor.intercept(context)

    stack = context.state.input.context_stack
    assert isinstance(stack, ContextStack)
    assert len(stack.layers) == 1
    assert isinstance(stack.layers[0], RuntimeInstructionsLayer)
    assert "ask the user how they would like to continue" in stack.layers[0].text
    assert stack.all_tools() == []
