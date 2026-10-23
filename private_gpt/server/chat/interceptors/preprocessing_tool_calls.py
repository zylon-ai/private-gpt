"""Lifecycle of the synthetic tool calls emitted by preprocessing interceptors.

Document and multimodal preprocessing surface their work to the client as a
``tool_use`` / ``tool_result`` pair per attachment. The stream contract is that
every ``tool_use`` is answered by exactly one ``tool_result``; a dangling
``tool_use`` breaks clients and, when ``return_type == "tool_result"``, produces
a history the model rejects. This helper owns that invariant so the interceptors
cannot forget it on any code path, including exceptions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.tools.events.adapters import ServerToolEventAdapter
from private_gpt.components.tools.tool_execution_outcome import (
    ToolExecutionError,
    ToolExecutionFailure,
    ToolExecutionSuccess,
)
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    to_llama_index_blocks,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )
    from private_gpt.events.models import (
        ResultContentBlockType,
        ToolResultBlock,
        ToolUseBlock,
    )

logger = logging.getLogger(__name__)

ReturnType = Literal["user_message", "tool_result"]
ToolContent = str | list["ResultContentBlockType"]
_EVENT_ADAPTER = ServerToolEventAdapter()


class PreprocessingToolCalls:
    """Track emitted preprocessing ``tool_use`` blocks and guarantee their results."""

    def __init__(
        self,
        context: ChatInterceptorContext,
        tool_name: str,
        return_type: ReturnType,
        default_error: str,
    ) -> None:
        self._context = context
        self._tool_name = tool_name
        self._return_type = return_type
        self._default_error = default_error
        self._pending: dict[Hashable, str] = {}
        self._completed: list[tuple[str, ToolContent, bool]] = []

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def start(self, key: Hashable, tool_input: dict[str, Any]) -> str:
        """Emit a ``tool_use`` block for ``key`` and remember it as pending."""
        tool_id = _EVENT_ADAPTER.new_tool_use_id()
        self._pending[key] = tool_id
        self._emit(
            _EVENT_ADAPTER.build_tool_use(
                tool_id=tool_id, tool_name=self._tool_name, tool_input=tool_input
            )
        )
        return tool_id

    def finish(
        self,
        key: Hashable,
        content: ToolContent | None,
        *,
        is_error: bool,
        error_detail: str | None = None,
    ) -> str:
        """Emit the ``tool_result`` answering the ``tool_use`` started for ``key``."""
        tool_id = self._pending.pop(key, None)
        if tool_id is None:
            tool_id = self.start(key, {"key": str(key)})
            self._pending.pop(key, None)
        resolved: ToolContent = content or error_detail or self._default_error
        outcome: ToolExecutionFailure | ToolExecutionSuccess = (
            ToolExecutionFailure(error=ToolExecutionError(message=str(resolved)))
            if is_error
            else ToolExecutionSuccess(
                content=(
                    resolved
                    if isinstance(resolved, list)
                    else [TextBlock(text=resolved)]
                )
            )
        )
        self._emit(
            _EVENT_ADAPTER.build_tool_result(tool_use_id=tool_id, outcome=outcome)
        )
        if self._return_type == "tool_result":
            self._completed.append((tool_id, resolved, is_error))
        return tool_id

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def fail_pending(self, error: BaseException) -> None:
        """Answer every still-pending ``tool_use`` with an error ``tool_result``."""
        detail = str(error) or error.__class__.__name__
        for key in list(self._pending):
            self.finish(key, None, is_error=True, error_detail=detail)

    def append_tool_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Append the assistant/tool message pair when carrying results in history."""
        if self._return_type != "tool_result" or not self._completed:
            return messages

        assistant_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id=tool_id, tool_name=self._tool_name, tool_kwargs={}
                    )
                    for tool_id, _, _ in self._completed
                ]
            },
        )
        tool_msgs: list[ChatMessage] = []
        for tool_id, content, _ in self._completed:
            kwargs = {
                "tool_call_id": tool_id,
                "tool_call_name": self._tool_name,
                "raw_output": content,
            }
            if isinstance(content, str):
                tool_msgs.append(
                    ChatMessage(
                        role=MessageRole.TOOL, content=content, additional_kwargs=kwargs
                    )
                )
            else:
                tool_msgs.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        blocks=to_llama_index_blocks(content),
                        additional_kwargs=kwargs,
                    )
                )
        return [*messages, assistant_msg, *tool_msgs]

    def _emit(self, block: ToolUseBlock | ToolResultBlock) -> None:
        start = RawContentBlockStartEvent(
            block_id=f"block_{uuid4().hex}", content_block=block
        )
        self._context.emit_event(start)
        self._context.emit_event(RawContentBlockStopEvent.from_start(start))
