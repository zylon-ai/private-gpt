import asyncio
import importlib
import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    TextBlock,
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.llms.openai.utils import O1_MODELS

from private_gpt.components.llm.custom.structured_mixin import StructuredChatMixin
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.model_discovery.url_utils import is_openai_api_base
from private_gpt.events.models import StopReasonEnum

if TYPE_CHECKING:
    from llama_index.llms.openai import (  # type: ignore[import-not-found,import-untyped]
        OpenAI as OpenAIBase,
    )

logger = logging.getLogger(__name__)


def _load_openai_base() -> type[Any]:
    try:
        return cast(
            type[Any], importlib.import_module("llama_index.llms.openai").OpenAI
        )
    except ImportError as e:
        from private_gpt.utils.dependencies import format_missing_dependency_message

        raise ImportError(
            format_missing_dependency_message(
                "OpenAI LLM",
                extras="llm-openai",
            )
        ) from e


if not TYPE_CHECKING:
    OpenAIBase = _load_openai_base()

_CUSTOM_REASONING_EFFORT_MAPPING = {
    ReasoningEffort.NONE: "none",
    ReasoningEffort.MAX: "xhigh",
    ReasoningEffort.XHIGH: "xhigh",
}


class PatchedOpenAILLM(StructuredChatMixin, OpenAIBase):  # type: ignore[misc]
    """Patched OpenAI LLM with fixes over the base llama_index implementation."""

    @staticmethod
    def _build_openai_response_format(
        structured_outputs: Any,
    ) -> dict[str, Any]:
        """Convert StructuredOutputsParams to an OpenAI response_format dict."""
        from private_gpt.components.llm.custom.base import StructuredOutputsParams

        if not isinstance(structured_outputs, StructuredOutputsParams):
            return {"type": "json_object"}

        if structured_outputs.json_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": structured_outputs.json_schema,
                },
            }

        # Fallback: regex / choice / grammar are not natively supported by OpenAI
        return {"type": "json_object"}

    def _normalize_chat_messages(
        self,
        messages: Sequence[ChatMessage],
    ) -> Sequence[ChatMessage]:
        """Normalize chat messages to ensure compatibility with OpenAI."""
        normalized = []
        for msg in messages:

            # OpenAI expects to have content even when it's None
            blocks = msg.blocks if msg.blocks else [TextBlock(text="")]

            # The tool_uses has to be in OpenAI format
            tool_calls = msg.additional_kwargs.get("tool_calls", [])
            if tool_calls:
                from openai.types.chat.chat_completion_chunk import (
                    ChoiceDeltaToolCall,
                    ChoiceDeltaToolCallFunction,
                )

                openai_tool_calls = []
                for i, tool_call in enumerate(tool_calls):

                    if isinstance(tool_call, ChoiceDeltaToolCall):
                        openai_tool_calls.append(tool_call)
                    elif isinstance(tool_call, ToolSelection):

                        openai_tool_calls.append(
                            ChoiceDeltaToolCall(
                                index=i,
                                id=tool_call.tool_id,
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=tool_call.tool_name,
                                    arguments=json.dumps(tool_call.tool_kwargs),
                                ),
                            )
                        )
                msg.additional_kwargs["tool_calls"] = openai_tool_calls

            normalized.append(
                ChatMessage(
                    role=msg.role,
                    blocks=blocks,
                    additional_kwargs=msg.additional_kwargs,
                )
            )
        return normalized

    def _normalize_openai_response(
        self,
        chat_response: ChatResponse,
    ) -> ChatResponse:
        """Normalize OpenAI-specific fields to the unified format."""
        from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

        raw = cast(ChatCompletionChunk, chat_response.raw)

        # Normalize usage keys
        additional_kwargs = chat_response.additional_kwargs
        if "prompt_tokens" in additional_kwargs:
            additional_kwargs["input_tokens"] = additional_kwargs.pop("prompt_tokens")
        if "completion_tokens" in additional_kwargs:
            additional_kwargs["output_tokens"] = additional_kwargs.pop(
                "completion_tokens"
            )
        additional_kwargs.pop("total_tokens", None)

        # Extract finish_reason and convert to stop_reason
        if raw is not None and len(raw.choices) > 0:
            finish_reason = raw.choices[0].finish_reason
            if finish_reason is not None:
                stop_reason = StopReasonEnum.convert_from_openai(finish_reason)
                if stop_reason is not None:
                    additional_kwargs["stop_reason"] = stop_reason
                    chat_response.message.additional_kwargs["stop_reason"] = stop_reason

        return chat_response

    def _extract_reasoning_content(
        self,
        chat_response: ChatResponse,
    ) -> ChatResponse:
        """Extract reasoning content from OpenAI response deltas."""
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            ChoiceDelta,
        )

        response = cast(ChatCompletionChunk, chat_response.raw)
        if len(response.choices) > 0:
            delta = response.choices[0].delta
        else:
            delta = ChoiceDelta()

        if delta is None:
            return chat_response

        # Extract reasoning_content for chain-of-thought streaming.
        # It's bugged in the latest LI version.
        # PR: https://github.com/run-llama/llama_index/pull/21220
        raw_reasoning = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        reasoning_delta = raw_reasoning if isinstance(raw_reasoning, str) else ""

        if reasoning_delta:
            chat_response.additional_kwargs["thinking_delta"] = reasoning_delta
            chat_response.message.additional_kwargs["thinking_delta"] = reasoning_delta

        return chat_response

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _ensure_valid_reasoning_effort(
        self, reasoning_effort: ReasoningEffort | str | None
    ) -> str | None:
        if reasoning_effort is None:
            return None

        reasoning_effort_enum: ReasoningEffort = (
            ReasoningEffort.from_str(reasoning_effort)
            if isinstance(reasoning_effort, str)
            else reasoning_effort
        )
        return _CUSTOM_REASONING_EFFORT_MAPPING.get(
            reasoning_effort_enum,
            reasoning_effort_enum.value,
        )

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        messages = [*messages]
        messages = self._normalize_chat_messages(messages)

        response = super()._stream_chat(
            messages=messages,
            **kwargs,
        )

        for chat_response in response:
            chat_response = self._extract_reasoning_content(chat_response)
            chat_response = self._normalize_openai_response(chat_response)
            yield chat_response

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        messages = [*messages]
        messages = self._normalize_chat_messages(messages)

        parent_gen = super()._astream_chat(messages=messages, **kwargs)

        def process_chat_response(chat_response: ChatResponse) -> ChatResponse:
            chat_response = self._extract_reasoning_content(chat_response)
            chat_response = self._normalize_openai_response(chat_response)
            return chat_response

        async def coro() -> ChatResponseAsyncGen:
            async for chat_response in await parent_gen:
                yield await asyncio.to_thread(process_chat_response, chat_response)

        return coro()

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        from private_gpt.components.llm.custom.base import SamplingParameters

        all_kwargs = super()._get_model_kwargs(**kwargs)
        is_real_openai = is_openai_api_base(self.api_base)

        if not is_real_openai:
            # Apply only supported sampling parameters for OpenAI-compatible APIs.
            for key in SamplingParameters.valid_keys():
                if key in kwargs and kwargs[key] is not None:
                    all_kwargs[key] = kwargs[key]

        # Parent only sets reasoning_effort for O1_MODELS; apply it for all models.
        if kwargs.get("reasoning_effort") is not None:
            all_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
        elif self.reasoning_effort is not None:
            all_kwargs["reasoning_effort"] = self.reasoning_effort

        if all_kwargs.get("reasoning_effort") is not None:
            all_kwargs["reasoning_effort"] = self._ensure_valid_reasoning_effort(
                all_kwargs["reasoning_effort"]
            )
        if is_real_openai:
            if self.model not in O1_MODELS:
                all_kwargs.pop("reasoning_effort", None)
        elif "reasoning_effort" not in all_kwargs:
            all_kwargs["reasoning_effort"] = "none"

        # Convert StructuredOutputsParams to OpenAI response_format.
        structured_outputs = all_kwargs.pop("structured_outputs", None)
        if structured_outputs is not None:
            all_kwargs["response_format"] = self._build_openai_response_format(
                structured_outputs
            )

        # Request usage in the last streaming chunk so input/output token counts
        # are available for normalization.
        if all_kwargs.get("stream") and "stream_options" not in all_kwargs:
            all_kwargs["stream_options"] = {"include_usage": True}

        return all_kwargs  # type: ignore[no-any-return]
