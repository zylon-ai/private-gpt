import importlib
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
    ToolCallBlock,
)

from private_gpt.components.llm.custom.structured_mixin import StructuredChatMixin
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.events.models import StopReasonEnum

if TYPE_CHECKING:
    from llama_index.llms.openai import (  # type: ignore[import-not-found,import-untyped]
        OpenAIResponses as OpenAIResponsesBase,
    )
    from openai.types.responses import ResponseFunctionToolCall

logger = logging.getLogger(__name__)


def _load_openai_responses_base() -> type[Any]:
    try:
        return cast(
            type[Any],
            importlib.import_module("llama_index.llms.openai").OpenAIResponses,
        )
    except (ImportError, AttributeError) as e:
        from private_gpt.utils.dependencies import format_missing_dependency_message

        raise ImportError(
            format_missing_dependency_message(
                "OpenAI Responses LLM",
                extras="llm-openai",
            )
        ) from e


if not TYPE_CHECKING:
    OpenAIResponsesBase = _load_openai_responses_base()

_RESPONSES_REASONING_EFFORT_MAP = {
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
    ReasoningEffort.MAX: "high",
}


class PatchedOpenAIResponsesLLM(StructuredChatMixin, OpenAIResponsesBase):  # type: ignore[misc]
    """Patched OpenAI Responses LLM.

    Adds reasoning, tools, and structured-output support.
    """

    @staticmethod
    def _build_responses_text_format(structured_outputs: Any) -> dict[str, Any] | None:
        """Convert StructuredOutputsParams to Responses API text.format dict."""
        from private_gpt.components.llm.custom.base import StructuredOutputsParams

        if not isinstance(structured_outputs, StructuredOutputsParams):
            return {"type": "json_object"}

        if structured_outputs.json_schema:
            return {
                "type": "json_schema",
                "name": "response",
                "strict": True,
                "schema": structured_outputs.json_schema,
            }

        return {"type": "json_object"}

    def _ensure_valid_reasoning_effort(
        self, reasoning_effort: "ReasoningEffort | str | None"
    ) -> dict[str, Any] | None:
        """Convert ReasoningEffort to the reasoning_options dict for Responses API."""
        if reasoning_effort is None:
            return None

        effort_enum: ReasoningEffort = (
            ReasoningEffort.from_str(reasoning_effort)
            if isinstance(reasoning_effort, str)
            else reasoning_effort
        )

        if effort_enum == ReasoningEffort.NONE:
            return None

        effort_str = _RESPONSES_REASONING_EFFORT_MAP.get(effort_enum, effort_enum.value)
        return {"effort": effort_str, "summary": "auto"}

    @staticmethod
    def _get_stop_reason_from_response(response: Any) -> StopReasonEnum:
        """Derive a unified StopReasonEnum from a Responses API Response object."""
        status = getattr(response, "status", None)

        if status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            reason = getattr(incomplete_details, "reason", None)
            match reason:
                case "max_output_tokens":
                    return StopReasonEnum.MAX_TOKENS
                case "content_filter":
                    return StopReasonEnum.REFUSAL
                case _:
                    return StopReasonEnum.END_TURN

        # Check output items for any function calls
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                return StopReasonEnum.TOOL_USE

        return StopReasonEnum.END_TURN

    def _normalize_completed_kwargs(
        self,
        additional_kwargs: dict[str, Any],
        response: Any,
        blocks: list[Any],
    ) -> dict[str, Any]:
        """Add input_tokens, output_tokens, and stop_reason to additional_kwargs."""
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if input_tokens is not None:
                additional_kwargs["input_tokens"] = input_tokens
            if output_tokens is not None:
                additional_kwargs["output_tokens"] = output_tokens

        # Prefer response-level stop reason; fall back to checking blocks
        stop_reason = self._get_stop_reason_from_response(response)
        if stop_reason == StopReasonEnum.END_TURN and any(
            isinstance(b, ToolCallBlock) for b in blocks
        ):
            stop_reason = StopReasonEnum.TOOL_USE

        additional_kwargs["stop_reason"] = stop_reason
        return additional_kwargs

    @staticmethod
    def _normalize_messages_for_responses_api(
        messages: Sequence[ChatMessage],
    ) -> Sequence[ChatMessage]:
        """Move tool calls from additional_kwargs into message.blocks for the Responses.

        After streaming, _handle_stream_chunk stores accumulated ToolSelection /
        ToolCallBlock objects in assistant_message.additional_kwargs["tool_calls"]
        while message.blocks stays empty.  to_openai_responses_message_dict uses the
        blocks path when blocks are present, so we move the tool calls there and remove
        the key to avoid the wrong serialization path (ToolSelection.model_dump()
        produces {"tool_id": …} instead of the required {"type": "function_call", …}).
        """
        from llama_index.core.llms.llm import ToolSelection

        for message in messages:
            if message.role != MessageRole.ASSISTANT:
                continue
            raw_tool_calls = message.additional_kwargs.get("tool_calls")
            if not raw_tool_calls:
                continue

            new_tool_blocks: list[ToolCallBlock] = []
            for tc in raw_tool_calls:
                if isinstance(tc, ToolCallBlock):
                    new_tool_blocks.append(tc)
                elif isinstance(tc, ToolSelection):
                    new_tool_blocks.append(
                        ToolCallBlock(
                            tool_call_id=tc.tool_id,
                            tool_name=tc.tool_name,
                            tool_kwargs=tc.tool_kwargs,
                        )
                    )

            if new_tool_blocks:
                non_tool_blocks = [
                    b for b in message.blocks if not isinstance(b, ToolCallBlock)
                ]
                message.blocks = non_tool_blocks + new_tool_blocks
                del message.additional_kwargs["tool_calls"]

        return messages

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[Any]:
        """Extract tool calls, falling back to additional_kwargs when blocks are empty.

        The base implementation only checks message.blocks, but _handle_stream_chunk
        never accumulates blocks — it stores ToolCallBlock objects in
        message.additional_kwargs["tool_calls"] instead.  We check both locations.
        """
        from llama_index.core.llms.llm import ToolSelection
        from llama_index.core.llms.utils import parse_partial_json

        # Non-streaming / ResponseCompletedEvent path: blocks populated directly
        tool_call_blocks = [
            b for b in response.message.blocks if isinstance(b, ToolCallBlock)
        ]

        # Streaming accumulation path: _handle_stream_chunk stores them here
        if not tool_call_blocks:
            raw = response.message.additional_kwargs.get("tool_calls", [])
            tool_call_blocks = [tc for tc in raw if isinstance(tc, ToolCallBlock)]

        if not tool_call_blocks:
            if error_on_no_tool_call:
                raise ValueError(
                    "Expected tool calls in response but found none. "
                    f"Message: {response.message}"
                )
            return []

        tool_selections = []
        for b in tool_call_blocks:
            # tool_kwargs may be a JSON string (Responses API) or already a dict
            raw_kwargs = b.tool_kwargs
            if isinstance(raw_kwargs, str):
                try:
                    argument_dict = parse_partial_json(raw_kwargs) or {}
                except Exception:
                    argument_dict = {}
            else:
                argument_dict = raw_kwargs or {}
            tool_selections.append(
                ToolSelection(
                    tool_id=b.tool_call_id or "",
                    tool_name=b.tool_name,
                    tool_kwargs=argument_dict,
                )
            )
        return tool_selections

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Handle reasoning_effort, structured_outputs, and BaseTool conversion."""
        # Pop our custom params before forwarding to the parent
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        structured_outputs = kwargs.pop("structured_outputs", None)
        raw_tools = kwargs.pop("tools", None) or []

        # Convert BaseTool objects to Responses API function-spec dicts
        from llama_index.core.tools.types import BaseTool

        converted_tools: list[Any] = []
        for tool in raw_tools:
            if isinstance(tool, BaseTool):
                spec: dict[str, Any] = {
                    "type": "function",
                    **tool.metadata.to_openai_tool(skip_length_check=True)["function"],
                }
                if self.strict:
                    spec["strict"] = True
                    spec.get("parameters", {})["additionalProperties"] = False
                converted_tools.append(spec)
            else:
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools

        model_kwargs = super()._get_model_kwargs(**kwargs)

        # Apply reasoning from effort kwarg (overrides class-level reasoning_options)
        reasoning_options = self._ensure_valid_reasoning_effort(reasoning_effort)
        if reasoning_options is not None:
            model_kwargs["reasoning"] = reasoning_options
            for param in (
                "top_p",
                "temperature",
                "presence_penalty",
                "frequency_penalty",
            ):
                model_kwargs.pop(param, None)

        # Map max_tokens → max_output_tokens if present
        if "max_tokens" in model_kwargs and "max_output_tokens" not in model_kwargs:
            model_kwargs["max_output_tokens"] = model_kwargs.pop("max_tokens")
        elif "max_tokens" in model_kwargs:
            model_kwargs.pop("max_tokens")

        # Apply structured outputs as Responses API text.format
        if structured_outputs is not None:
            text_format = self._build_responses_text_format(structured_outputs)
            if text_format is not None:
                model_kwargs["text"] = {"format": text_format}

        return model_kwargs

    def _stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Stream chat with thinking_delta, stop_reason, and usage normalization."""
        from llama_index.llms.openai import (
            OpenAIResponses as _OpenAIResponsesBase,
        )
        from llama_index.llms.openai.utils import to_openai_message_dicts
        from openai.types.responses import (
            ResponseCompletedEvent,
            ResponseReasoningSummaryTextDeltaEvent,
            ResponseReasoningTextDeltaEvent,
        )

        messages = self._normalize_messages_for_responses_api(messages)
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        def gen() -> ChatResponseGen:
            built_in_tool_calls: list[Any] = []
            additional_kwargs: dict[str, Any] = {"built_in_tool_calls": []}
            current_tool_call: ResponseFunctionToolCall | None = None
            local_previous_response_id = self._previous_response_id

            for event in self._client.responses.create(
                input=message_dicts,  # type: ignore[arg-type]
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                # Reasoning deltas: put thinking_delta in ChatMessage.additional_kwargs
                # so _extract_reasoning (which checks chunk.message.additional_kwargs)
                # can pick it up.  Do NOT set it in the outer additional_kwargs dict —
                # that dict persists across iterations and would corrupt later chunks.
                if isinstance(
                    event,
                    ResponseReasoningSummaryTextDeltaEvent
                    | ResponseReasoningTextDeltaEvent,
                ):
                    if event.delta:
                        yield ChatResponse(
                            message=ChatMessage(
                                role="assistant",
                                blocks=[],
                                additional_kwargs={"thinking_delta": event.delta},
                            ),
                            delta="",
                            raw=event,
                            additional_kwargs=dict(additional_kwargs),
                        )
                    continue

                (
                    blocks,
                    built_in_tool_calls,
                    additional_kwargs,
                    current_tool_call,
                    local_previous_response_id,
                    delta,
                ) = _OpenAIResponsesBase.process_response_event(
                    event=event,  # type: ignore[arg-type]
                    built_in_tool_calls=built_in_tool_calls,
                    additional_kwargs=additional_kwargs,
                    current_tool_call=current_tool_call,
                    track_previous_responses=self.track_previous_responses,
                    previous_response_id=local_previous_response_id,
                )

                if (
                    self.track_previous_responses
                    and local_previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = local_previous_response_id

                if built_in_tool_calls:
                    additional_kwargs["built_in_tool_calls"] = built_in_tool_calls

                if isinstance(event, ResponseCompletedEvent):
                    additional_kwargs = self._normalize_completed_kwargs(
                        additional_kwargs,
                        event.response,
                        blocks,
                    )
                    stop_reason = additional_kwargs.get("stop_reason")
                    # Put ToolCallBlocks in message.additional_kwargs["tool_calls"]
                    # so _handle_stream_chunk accumulates them and
                    # get_tool_calls_from_response can find them via the fallback path
                    # (blocks are never accumulated by _handle_stream_chunk).
                    tool_call_blocks = [
                        b for b in blocks if isinstance(b, ToolCallBlock)
                    ]
                    message_extra: dict[str, Any] = {"stop_reason": stop_reason}
                    if tool_call_blocks:
                        message_extra["tool_calls"] = tool_call_blocks
                    message = ChatMessage(
                        role="assistant",
                        blocks=blocks,
                        additional_kwargs=message_extra,
                    )
                    yield ChatResponse(
                        message=message,
                        delta=delta,
                        raw=event,
                        additional_kwargs=additional_kwargs,
                    )
                    continue

                yield ChatResponse(
                    message=ChatMessage(role="assistant", blocks=blocks),
                    delta=delta,
                    raw=event,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async stream chat normalizing thinking_delta, stop_reason, and usage."""
        from llama_index.llms.openai import (
            OpenAIResponses as _OpenAIResponsesBase,
        )
        from llama_index.llms.openai.utils import to_openai_message_dicts
        from openai.types.responses import (
            ResponseCompletedEvent,
            ResponseReasoningSummaryTextDeltaEvent,
            ResponseReasoningTextDeltaEvent,
        )

        messages = self._normalize_messages_for_responses_api(messages)
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        async def gen() -> ChatResponseAsyncGen:
            built_in_tool_calls: list[Any] = []
            additional_kwargs: dict[str, Any] = {"built_in_tool_calls": []}
            current_tool_call: ResponseFunctionToolCall | None = None
            local_previous_response_id = self._previous_response_id

            response_stream = await self._aclient.responses.create(
                input=message_dicts,  # type: ignore[arg-type]
                stream=True,
                **self._get_model_kwargs(**kwargs),
            )

            async for event in response_stream:  # type: ignore[union-attr]
                if isinstance(
                    event,
                    ResponseReasoningSummaryTextDeltaEvent
                    | ResponseReasoningTextDeltaEvent,
                ):
                    if event.delta:
                        yield ChatResponse(
                            message=ChatMessage(
                                role="assistant",
                                blocks=[],
                                additional_kwargs={"thinking_delta": event.delta},
                            ),
                            delta="",
                            raw=event,
                            additional_kwargs=dict(additional_kwargs),
                        )
                    continue

                (
                    blocks,
                    built_in_tool_calls,
                    additional_kwargs,
                    current_tool_call,
                    local_previous_response_id,
                    delta,
                ) = _OpenAIResponsesBase.process_response_event(
                    event=event,  # type: ignore[arg-type]
                    built_in_tool_calls=built_in_tool_calls,
                    additional_kwargs=additional_kwargs,
                    current_tool_call=current_tool_call,
                    track_previous_responses=self.track_previous_responses,
                    previous_response_id=local_previous_response_id,
                )

                if (
                    self.track_previous_responses
                    and local_previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = local_previous_response_id

                if built_in_tool_calls:
                    additional_kwargs["built_in_tool_calls"] = built_in_tool_calls

                if isinstance(event, ResponseCompletedEvent):
                    additional_kwargs = self._normalize_completed_kwargs(
                        additional_kwargs,
                        event.response,
                        blocks,
                    )
                    stop_reason = additional_kwargs.get("stop_reason")
                    tool_call_blocks = [
                        b for b in blocks if isinstance(b, ToolCallBlock)
                    ]
                    message_extra: dict[str, Any] = {"stop_reason": stop_reason}
                    if tool_call_blocks:
                        message_extra["tool_calls"] = tool_call_blocks
                    message = ChatMessage(
                        role="assistant",
                        blocks=blocks,
                        additional_kwargs=message_extra,
                    )
                    yield ChatResponse(
                        message=message,
                        delta=delta,
                        raw=event,
                        additional_kwargs=additional_kwargs,
                    )
                    continue

                yield ChatResponse(
                    message=ChatMessage(role="assistant", blocks=blocks),
                    delta=delta,
                    raw=event,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Non-streaming chat with usage and stop_reason normalization."""
        messages = self._normalize_messages_for_responses_api(messages)
        chat_response: ChatResponse = super()._chat(messages, **kwargs)

        usage = chat_response.additional_kwargs.get("usage")
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if input_tokens is not None:
                chat_response.additional_kwargs["input_tokens"] = input_tokens
            if output_tokens is not None:
                chat_response.additional_kwargs["output_tokens"] = output_tokens

        raw_response = getattr(chat_response, "raw", None)
        if raw_response is not None:
            stop_reason = self._get_stop_reason_from_response(raw_response)
        else:
            has_tool_calls = any(
                isinstance(b, ToolCallBlock) for b in chat_response.message.blocks
            )
            stop_reason = (
                StopReasonEnum.TOOL_USE if has_tool_calls else StopReasonEnum.END_TURN
            )

        chat_response.additional_kwargs["stop_reason"] = stop_reason
        chat_response.message.additional_kwargs["stop_reason"] = stop_reason

        return chat_response

    async def _achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async non-streaming chat with usage and stop_reason normalization."""
        messages = self._normalize_messages_for_responses_api(messages)
        chat_response: ChatResponse = await super()._achat(messages, **kwargs)

        usage = chat_response.additional_kwargs.get("usage")
        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            if input_tokens is not None:
                chat_response.additional_kwargs["input_tokens"] = input_tokens
            if output_tokens is not None:
                chat_response.additional_kwargs["output_tokens"] = output_tokens

        raw_response = getattr(chat_response, "raw", None)
        if raw_response is not None:
            stop_reason = self._get_stop_reason_from_response(raw_response)
        else:
            has_tool_calls = any(
                isinstance(b, ToolCallBlock) for b in chat_response.message.blocks
            )
            stop_reason = (
                StopReasonEnum.TOOL_USE if has_tool_calls else StopReasonEnum.END_TURN
            )

        chat_response.additional_kwargs["stop_reason"] = stop_reason
        chat_response.message.additional_kwargs["stop_reason"] = stop_reason

        return chat_response
