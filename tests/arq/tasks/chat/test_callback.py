from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.arq.tasks.chat.callback import resume_chat_callback
from private_gpt.components.engines.chat.execution_scheduler import (
    ChatExecutionSchedulerFactory,
)
from private_gpt.components.engines.chat.models.execution_hooks import ExecutionHooks
from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from private_gpt.events.models import TextBlock


@pytest.mark.anyio
async def test_resume_chat_callback_sends_error_tool_result_to_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = MagicMock()
    scheduler.callback = AsyncMock()
    scheduler_factory = MagicMock()
    scheduler_factory.get.return_value = scheduler
    injector = MagicMock()
    injector.get.return_value = scheduler_factory
    monkeypatch.setattr(
        "private_gpt.di.get_global_injector",
        MagicMock(return_value=injector),
    )
    request = ToolExecutionRequest.model_validate(
        {
            "tool_id": "semantic-search-1",
            "tool_name": "semantic_search",
            "tool_kwargs": {},
            "tool_spec": {
                "name": "semantic_search",
                "runtime": "server",
                "input_schema": {},
            },
            "context": {"correlation_id": "chat-1"},
            "hooks": ExecutionHooks(),
        }
    )
    response = ToolExecutionResponse(
        tool_name="semantic_search",
        tool_id="semantic-search-1",
        result_content=[TextBlock(text="query: Field required")],
        is_error=True,
        tool_message={
            "role": "tool",
            "content": "query: Field required",
            "additional_kwargs": {
                "tool_call_id": "semantic-search-1",
                "tool_call_name": "semantic_search",
                "tool_call_args": {},
                "raw_output": "query: Field required",
            },
        },
    )

    await resume_chat_callback(request=request, response=response)

    injector.get.assert_called_once_with(ChatExecutionSchedulerFactory)
    scheduler.callback.assert_awaited_once_with(
        execution_id="chat-1",
        tool_id="semantic-search-1",
        result=response.model_dump(mode="json"),
    )
