from collections.abc import Sequence
from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import MockLLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools import BaseTool


class FunctionCallingLLMMock(MockLLM, FunctionCallingLLM):
    """Mock LLM that can be used for testing purposes."""

    max_tokens: int | None = None
    is_function_calling_model: bool | None = None

    _user_msg: str | ChatMessage | None = None
    _tools: Sequence[BaseTool] | None = None
    _run_tool: bool = False

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens or -1,
            is_function_calling_model=self.is_function_calling_model or True,
        )

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if user_msg:
            self._user_msg = (
                user_msg.content if isinstance(user_msg, ChatMessage) else str(user_msg)
            )
        self._tools = tools

        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        self._run_tool = False
        return {
            "messages": messages,
            "tools": tools,
        }

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolSelection]:
        if self._tools and not self._run_tool:
            self._run_tool = True
            return [
                ToolSelection(
                    tool_id=tool.metadata.name or "",
                    tool_name=tool.metadata.name or "",
                    tool_kwargs={
                        "query": self._user_msg,
                    },
                )
                for tool in self._tools[:1]
            ]

        return []
