from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM

from private_gpt.components.llm.decorators.reasoning_budget import (
    _EFFORT_BUDGET_RATIO,
    _build_continue_messages,
    _effective_reasoning,
    _extract_budget,
    _is_truncated_thinking,
    _merge_responses,
    _phase2_kwargs,
    _resolve_budget,
    _resolve_stop_reason,
    with_reasoning_budget_achat,
    with_reasoning_budget_astream_chat,
    with_reasoning_budget_chat,
    with_reasoning_budget_stream_chat,
)
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.events.models import StopReasonEnum
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


def _make_metadata(num_output: int = 1000) -> LLMMetadata:
    return LLMMetadata(num_output=num_output)


def _make_mock_llm(metadata: LLMMetadata | None = None) -> FunctionCallingLLM:
    meta = metadata or _make_metadata()
    mock = get_mock_function_calling_llm()
    mock.metadata = meta
    mock.get_metadata = MagicMock(return_value=meta)
    return mock


def _chat_response(
    content: str = "",
    stop_reason: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    thinking: str | None = None,
) -> ChatResponse:
    msg_kwargs: dict[str, Any] = {}
    if thinking:
        msg_kwargs["thinking"] = thinking
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            additional_kwargs=msg_kwargs,
        ),
        additional_kwargs={
            "stop_reason": stop_reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )


def _truncated_thinking_response(thinking: str = "<think>partial") -> ChatResponse:
    return _chat_response(
        content="",
        stop_reason=StopReasonEnum.MAX_TOKENS,
        input_tokens=10,
        output_tokens=50,
        thinking=thinking,
    )


def _sync_gen(*responses: ChatResponse) -> ChatResponseGen:
    def _inner() -> Generator[ChatResponse, None, None]:
        yield from responses

    return _inner()


async def _async_gen(*responses: ChatResponse) -> ChatResponseAsyncGen:
    async def _inner() -> AsyncGenerator[ChatResponse, None]:
        for r in responses:
            yield r

    return _inner()


class _DecoratedMockLLM:
    """Test helper that applies reasoning_effort budget decorators over a mock LLM."""

    def __init__(self, mock_llm: FunctionCallingLLM) -> None:
        self._mock = mock_llm

    @property
    def metadata(self) -> LLMMetadata:
        return self._mock.metadata

    def get_metadata(self, **kwargs: Any) -> LLMMetadata:
        return self._mock.get_metadata(**kwargs)

    @property
    def allow_reasoning_budget(self) -> bool:
        return getattr(self._mock, "allow_reasoning_budget", True)

    @with_reasoning_budget_chat
    def chat(
        self,
        messages,
        tools=None,
        reasoning_effort=ReasoningEffort.NONE,
        **kwargs,
    ) -> ChatResponse:
        return self._mock.chat(
            messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
        )

    @with_reasoning_budget_stream_chat
    def stream_chat(
        self,
        messages,
        tools=None,
        reasoning_effort=ReasoningEffort.NONE,
        **kwargs,
    ) -> ChatResponseGen:
        return self._mock.stream_chat(
            messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
        )

    @with_reasoning_budget_achat
    async def achat(
        self,
        messages,
        tools=None,
        reasoning_effort=ReasoningEffort.NONE,
        **kwargs,
    ) -> ChatResponse:
        return await self._mock.achat(
            messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
        )

    @with_reasoning_budget_astream_chat
    async def astream_chat(
        self,
        messages,
        tools=None,
        reasoning_effort=ReasoningEffort.NONE,
        **kwargs,
    ) -> ChatResponseAsyncGen:
        return await self._mock.astream_chat(
            messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
        )


class TestResolveStopReason:
    def test_from_additional_kwargs(self) -> None:
        assert (
            _resolve_stop_reason(_chat_response(stop_reason=StopReasonEnum.END_TURN))
            == StopReasonEnum.END_TURN
        )

    def test_falls_back_to_message_kwargs(self) -> None:
        r = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="hi",
                additional_kwargs={"stop_reason": StopReasonEnum.END_TURN},
            ),
            additional_kwargs={},
        )
        assert _resolve_stop_reason(r) == StopReasonEnum.END_TURN

    def test_returns_none_when_absent(self) -> None:
        assert _resolve_stop_reason(_chat_response()) is None


class TestIsTruncatedThinking:
    def test_true_when_max_tokens_and_no_content(self) -> None:
        assert (
            _is_truncated_thinking(
                _chat_response(stop_reason=StopReasonEnum.MAX_TOKENS)
            )
            is True
        )

    def test_false_when_has_content(self) -> None:
        assert (
            _is_truncated_thinking(
                _chat_response(stop_reason=StopReasonEnum.MAX_TOKENS, content="hi")
            )
            is False
        )

    def test_false_when_stop_reason_is_stop(self) -> None:
        assert (
            _is_truncated_thinking(_chat_response(stop_reason=StopReasonEnum.END_TURN))
            is False
        )

    def test_false_when_has_tool_calls(self) -> None:
        r = ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={"tool_calls": [MagicMock()]},
            ),
            additional_kwargs={"stop_reason": StopReasonEnum.MAX_TOKENS},
        )
        assert _is_truncated_thinking(r) is False

    def test_false_when_no_stop_reason(self) -> None:
        assert _is_truncated_thinking(_chat_response()) is False

    def test_whitespace_only_content_counts_as_empty(self) -> None:
        assert (
            _is_truncated_thinking(
                _chat_response(stop_reason=StopReasonEnum.MAX_TOKENS, content="   ")
            )
            is True
        )


class TestMergeResponses:
    def test_content_from_phase2(self) -> None:
        p1 = _chat_response(thinking="t", input_tokens=10, output_tokens=50)
        p2 = _chat_response(content="answer", input_tokens=60, output_tokens=30)
        assert _merge_responses(p1, p2).message.content == "answer"

    def test_thinking_from_phase1(self) -> None:
        p1 = _chat_response(thinking="deep thought", input_tokens=5, output_tokens=20)
        p2 = _chat_response(content="done", input_tokens=25, output_tokens=10)
        assert (
            _merge_responses(p1, p2).message.additional_kwargs["thinking"]
            == "deep thought"
        )

    def test_usage_summed(self) -> None:
        merged = _merge_responses(
            _chat_response(input_tokens=10, output_tokens=50),
            _chat_response(content="x", input_tokens=60, output_tokens=30),
        )
        assert merged.additional_kwargs["input_tokens"] == 70
        assert merged.additional_kwargs["output_tokens"] == 80

    def test_no_thinking_in_phase1_not_added(self) -> None:
        merged = _merge_responses(_chat_response(), _chat_response(content="ok"))
        assert "thinking" not in merged.message.additional_kwargs

    def test_phase2_extra_kwargs_preserved(self) -> None:
        p2 = _chat_response(content="ok")
        p2.message.additional_kwargs["custom"] = "val"
        assert (
            _merge_responses(_chat_response(), p2).message.additional_kwargs["custom"]
            == "val"
        )


class TestExtractBudget:
    def test_extracts_and_removes_key(self) -> None:
        budget, _clean = _extract_budget(reasoning_budget=512, temp=0.7)
        assert budget == 512
        assert "reasoning_budget" not in _clean
        assert _clean["temp"] == 0.7

    def test_returns_none_when_absent(self) -> None:
        budget, _ = _extract_budget(temp=0.7)
        assert budget is None

    def test_casts_to_int(self) -> None:
        budget, _ = _extract_budget(reasoning_budget=256)
        assert budget == 256

    def test_empty_kwargs(self) -> None:
        budget, clean = _extract_budget()
        assert budget is None
        assert clean == {}


class TestEffectiveReasoning:
    def test_zero_budget_disables(self) -> None:
        assert _effective_reasoning(ReasoningEffort.HIGH, 0) == ReasoningEffort.NONE

    def test_none_budget_preserves(self) -> None:
        assert _effective_reasoning(ReasoningEffort.HIGH, None) == ReasoningEffort.HIGH

    def test_positive_budget_preserves(self) -> None:
        assert _effective_reasoning(ReasoningEffort.LOW, 100) == ReasoningEffort.LOW

    def test_already_none_unchanged(self) -> None:
        assert _effective_reasoning(ReasoningEffort.NONE, None) == ReasoningEffort.NONE


class TestBuildContinueMessages:
    def test_appends_phase1_message(self) -> None:
        messages = [ChatMessage(role=MessageRole.USER, content="hello")]
        phase1 = _chat_response(thinking="partial")
        result = _build_continue_messages(messages, phase1)
        assert len(result) == 2
        assert result[-1] is phase1.message

    def test_original_not_mutated(self) -> None:
        messages = [ChatMessage(role=MessageRole.USER, content="hello")]
        _build_continue_messages(messages, _chat_response())
        assert len(messages) == 1


class TestResolveBudget:
    def setup_method(self) -> None:
        self.llm = _make_mock_llm(metadata=_make_metadata(num_output=1000))

    def test_explicit_takes_priority(self) -> None:
        assert _resolve_budget(self.llm, 300, ReasoningEffort.LOW) == 300

    def test_explicit_overrides_effort(self) -> None:
        assert _resolve_budget(self.llm, 99, ReasoningEffort.HIGH) == 99

    def test_auto_low(self) -> None:
        assert _resolve_budget(self.llm, None, ReasoningEffort.LOW) == int(
            1000 * _EFFORT_BUDGET_RATIO[ReasoningEffort.LOW]
        )

    def test_auto_medium(self) -> None:
        assert _resolve_budget(self.llm, None, ReasoningEffort.MEDIUM) == int(
            1000 * _EFFORT_BUDGET_RATIO[ReasoningEffort.MEDIUM]
        )

    def test_auto_high(self) -> None:
        assert _resolve_budget(self.llm, None, ReasoningEffort.HIGH) == int(
            1000 * _EFFORT_BUDGET_RATIO[ReasoningEffort.HIGH]
        )

    def test_none_effort_returns_none(self) -> None:
        assert _resolve_budget(self.llm, None, ReasoningEffort.NONE) is None

    def test_returns_none_when_total_zero(self) -> None:
        self.llm.get_metadata.return_value = _make_metadata(num_output=0)
        assert _resolve_budget(self.llm, None, ReasoningEffort.HIGH) is None

    def test_minimum_budget_is_1(self) -> None:
        self.llm.get_metadata.return_value = _make_metadata(num_output=1)
        assert _resolve_budget(self.llm, None, ReasoningEffort.LOW) == 1


class TestPhase2Kwargs:
    def setup_method(self) -> None:
        self.llm = _make_mock_llm(metadata=_make_metadata(num_output=1000))

    def test_remaining_tokens_set(self) -> None:
        max_tokens = self.llm.metadata.num_output
        assert _phase2_kwargs(max_tokens, 300)["max_tokens"] == 700

    def test_passthrough_when_total_unknown(self) -> None:
        result = _phase2_kwargs(0, 300, temp=0.5)
        assert "max_tokens" not in result

    def test_existing_kwargs_preserved(self) -> None:
        max_tokens = self.llm.metadata.num_output
        result = _phase2_kwargs(max_tokens, 200, temp=0.9)
        assert result["temp"] == 0.9
        assert result["max_tokens"] == 800

    def test_minimum_remaining_is_1(self) -> None:
        assert _phase2_kwargs(100, 99)["max_tokens"] == 1


class TestChat:
    def setup_method(self) -> None:
        self.llm = _make_mock_llm(metadata=_make_metadata(num_output=1000))
        self.wrapper = _DecoratedMockLLM(self.llm)
        self.messages = [ChatMessage(role=MessageRole.USER, content="hi")]

    def test_passthrough_reasoning_none(self) -> None:
        expected = _chat_response(content="direct")
        self.llm.chat.return_value = expected
        assert (
            self.wrapper.chat(self.messages, reasoning_effort=ReasoningEffort.NONE)
            is expected
        )
        self.llm.chat.assert_called_once()

    def test_passthrough_budget_zero(self) -> None:
        self.llm.chat.return_value = _chat_response(content="ok")
        self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, reasoning_budget=0
        )
        assert self.llm.chat.call_args[1]["reasoning_effort"] == ReasoningEffort.NONE

    def test_single_phase_not_truncated(self) -> None:
        response = _chat_response(content="answer", stop_reason=StopReasonEnum.END_TURN)
        self.llm.chat.return_value = response
        result = self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert self.llm.chat.call_count == 1
        assert result is response

    def test_two_phase_truncated(self) -> None:
        p1 = _truncated_thinking_response()
        p2 = _chat_response(content="final", input_tokens=60, output_tokens=30)
        self.llm.chat.side_effect = [p1, p2]
        result = self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert self.llm.chat.call_count == 2
        assert result.message.content == "final"
        assert result.additional_kwargs["input_tokens"] == 70
        assert result.additional_kwargs["output_tokens"] == 80

    def test_phase1_max_tokens_is_budget(self) -> None:
        self.llm.chat.return_value = _chat_response(
            content="ok", stop_reason=StopReasonEnum.END_TURN
        )
        self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=300
        )
        assert self.llm.chat.call_args_list[0][1]["max_tokens"] == 300

    def test_phase2_max_tokens_is_remaining(self) -> None:
        self.llm.chat.side_effect = [
            _truncated_thinking_response(),
            _chat_response(content="done"),
        ]
        self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert self.llm.chat.call_args_list[1][1]["max_tokens"] == 800

    def test_phase2_reasoning_none(self) -> None:
        self.llm.chat.side_effect = [
            _truncated_thinking_response(),
            _chat_response(content="done"),
        ]
        self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, reasoning_budget=200
        )
        assert (
            self.llm.chat.call_args_list[1][1]["reasoning_effort"]
            == ReasoningEffort.NONE
        )

    def test_phase2_messages_include_phase1_assistant(self) -> None:
        p1 = _truncated_thinking_response()
        self.llm.chat.side_effect = [p1, _chat_response(content="done")]
        self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert self.llm.chat.call_args_list[1][0][0][-1] is p1.message

    def test_auto_budget_from_effort(self) -> None:
        self.llm.chat.return_value = _chat_response(
            content="ok", stop_reason=StopReasonEnum.END_TURN
        )
        self.wrapper.chat(self.messages, reasoning_effort=ReasoningEffort.MEDIUM)
        expected = int(1000 * _EFFORT_BUDGET_RATIO[ReasoningEffort.MEDIUM])
        assert self.llm.chat.call_args_list[0][1]["max_tokens"] == expected

    def test_thinking_merged(self) -> None:
        p1 = _truncated_thinking_response(thinking="my thoughts")
        self.llm.chat.side_effect = [p1, _chat_response(content="done")]
        result = self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert result.message.additional_kwargs["thinking"] == "my thoughts"

    def test_reasoning_budget_kwarg_not_forwarded(self) -> None:
        self.llm.chat.return_value = _chat_response(
            content="ok", stop_reason=StopReasonEnum.END_TURN
        )
        self.wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert "reasoning_budget" not in self.llm.chat.call_args[1]


class TestStreamChat:
    def setup_method(self) -> None:
        self.llm = _make_mock_llm(metadata=_make_metadata(num_output=1000))
        self.wrapper = _DecoratedMockLLM(self.llm)
        self.messages = [ChatMessage(role=MessageRole.USER, content="hi")]

    def test_passthrough_reasoning_none(self) -> None:
        self.llm.stream_chat.return_value = _sync_gen(_chat_response(content="x"))
        assert (
            len(
                list(
                    self.wrapper.stream_chat(
                        self.messages, reasoning_effort=ReasoningEffort.NONE
                    )
                )
            )
            == 1
        )

    def test_single_phase_all_items_yielded(self) -> None:
        items = [_chat_response(content=f"t{i}") for i in range(3)]
        self.llm.stream_chat.return_value = _sync_gen(*items)
        result = list(
            self.wrapper.stream_chat(
                self.messages,
                reasoning_effort=ReasoningEffort.LOW,
                reasoning_budget=200,
            )
        )
        assert len(result) == 3
        assert self.llm.stream_chat.call_count == 1

    def test_two_phase_last_item_merged(self) -> None:
        p1 = [
            _chat_response(content=""),
            _truncated_thinking_response(thinking="thought"),
        ]
        p2 = [
            _chat_response(content="partial"),
            _chat_response(content="full", input_tokens=60, output_tokens=30),
        ]
        self.llm.stream_chat.side_effect = [_sync_gen(*p1), _sync_gen(*p2)]
        result = list(
            self.wrapper.stream_chat(
                self.messages,
                reasoning_effort=ReasoningEffort.LOW,
                reasoning_budget=200,
            )
        )
        assert self.llm.stream_chat.call_count == 2
        last = result[-1]
        assert last.message.content == "full"
        assert last.additional_kwargs["input_tokens"] == 70
        assert last.message.additional_kwargs.get("thinking") == "thought"

    def test_phase2_intermediates_yielded_unmerged(self) -> None:
        self.llm.stream_chat.side_effect = [
            _sync_gen(_truncated_thinking_response()),
            _sync_gen(_chat_response(content="part1"), _chat_response(content="part2")),
        ]
        result = list(
            self.wrapper.stream_chat(
                self.messages,
                reasoning_effort=ReasoningEffort.LOW,
                reasoning_budget=200,
            )
        )
        assert result[1].message.content == "part1"
        assert result[2].message.content == "part2"

    def test_passthrough_budget_zero(self) -> None:
        self.llm.stream_chat.return_value = _sync_gen(_chat_response(content="x"))
        list(
            self.wrapper.stream_chat(
                self.messages, reasoning_effort=ReasoningEffort.HIGH, reasoning_budget=0
            )
        )
        assert (
            self.llm.stream_chat.call_args[1]["reasoning_effort"]
            == ReasoningEffort.NONE
        )


class TestAchat:
    def setup_method(self) -> None:
        self.llm = _make_mock_llm(metadata=_make_metadata(num_output=1000))
        self.wrapper = _DecoratedMockLLM(self.llm)
        self.messages = [ChatMessage(role=MessageRole.USER, content="hi")]

    async def test_passthrough_reasoning_none(self) -> None:
        expected = _chat_response(content="direct")

        async def _achat(*a: Any, **kw: Any) -> ChatResponse:
            return expected

        self.llm.achat = _achat
        assert (
            await self.wrapper.achat(
                self.messages, reasoning_effort=ReasoningEffort.NONE
            )
            is expected
        )

    async def test_single_phase_not_truncated(self) -> None:
        response = _chat_response(content="answer", stop_reason=StopReasonEnum.END_TURN)
        calls = 0

        async def _achat(*a: Any, **kw: Any) -> ChatResponse:
            nonlocal calls
            calls += 1
            return response

        self.llm.achat = _achat
        result = await self.wrapper.achat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert calls == 1
        assert result is response

    async def test_two_phase_truncated(self) -> None:
        p1 = _truncated_thinking_response(thinking="thought")
        p2 = _chat_response(content="final", input_tokens=60, output_tokens=30)
        resps, calls = [p1, p2], 0

        async def _achat(*a: Any, **kw: Any) -> ChatResponse:
            nonlocal calls
            r = resps[calls]
            calls += 1
            return r

        self.llm.achat = _achat
        result = await self.wrapper.achat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        assert calls == 2
        assert result.message.content == "final"
        assert result.message.additional_kwargs["thinking"] == "thought"
        assert result.additional_kwargs["input_tokens"] == 70

    async def test_passthrough_budget_zero(self) -> None:
        received: list[Any] = []

        async def _achat(*a: Any, **kw: Any) -> ChatResponse:
            received.append(kw.get("reasoning_effort"))
            return _chat_response(content="ok")

        self.llm.achat = _achat
        await self.wrapper.achat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, reasoning_budget=0
        )
        assert received[0] == ReasoningEffort.NONE

    async def test_phase1_and_phase2_max_tokens(self) -> None:
        p1 = _truncated_thinking_response()
        p2 = _chat_response(content="done")
        received: list[Any] = []

        async def _achat(*a: Any, **kw: Any) -> ChatResponse:
            received.append(kw.get("max_tokens"))
            return p1 if len(received) == 1 else p2

        self.llm.achat = _achat
        await self.wrapper.achat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=300
        )
        assert received[0] == 300
        assert received[1] == 700


class TestAstreamChat:
    def setup_method(self) -> None:
        self.llm = _make_mock_llm(metadata=_make_metadata(num_output=1000))
        self.wrapper = _DecoratedMockLLM(self.llm)
        self.messages = [ChatMessage(role=MessageRole.USER, content="hi")]

    async def test_passthrough_reasoning_none(self) -> None:
        async def _astream(*a: Any, **kw: Any):
            return await _async_gen(_chat_response(content="x"))

        self.llm.astream_chat = _astream
        gen = await self.wrapper.astream_chat(
            self.messages, reasoning_effort=ReasoningEffort.NONE
        )
        assert len([i async for i in gen]) == 1

    async def test_two_phase_last_item_merged(self) -> None:
        p1_items = [_truncated_thinking_response(thinking="thought")]
        p2_items = [
            _chat_response(content="mid"),
            _chat_response(content="end", input_tokens=60, output_tokens=30),
        ]
        calls = 0

        async def _astream(*a: Any, **kw: Any):
            nonlocal calls
            items = p1_items if calls == 0 else p2_items
            calls += 1
            return await _async_gen(*items)

        self.llm.astream_chat = _astream
        gen = await self.wrapper.astream_chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW, reasoning_budget=200
        )
        result = [i async for i in gen]
        assert calls == 2
        last = result[-1]
        assert last.message.content == "end"
        assert last.message.additional_kwargs.get("thinking") == "thought"
        assert last.additional_kwargs["input_tokens"] == 70

    async def test_single_phase_no_second_call(self) -> None:
        calls = 0

        async def _astream(*a: Any, **kw: Any):
            nonlocal calls
            calls += 1
            return await _async_gen(
                _chat_response(content="done", stop_reason=StopReasonEnum.END_TURN)
            )

        self.llm.astream_chat = _astream
        gen = await self.wrapper.astream_chat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, reasoning_budget=400
        )
        result = [i async for i in gen]
        assert calls == 1
        assert len(result) == 1

    async def test_passthrough_budget_zero(self) -> None:
        received: list[Any] = []

        async def _astream(*a: Any, **kw: Any):
            received.append(kw.get("reasoning_effort"))
            return await _async_gen(_chat_response(content="x"))

        self.llm.astream_chat = _astream
        gen = await self.wrapper.astream_chat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, reasoning_budget=0
        )
        [i async for i in gen]
        assert received[0] == ReasoningEffort.NONE


class TestEnableReasoningBudgetFlag:
    """Test that enable_reasoning_budget flag properly disables budget logic."""

    def setup_method(self) -> None:
        self.messages = [ChatMessage(role=MessageRole.USER, content="test")]
        self.llm = _make_mock_llm(_make_metadata(1000))

    def test_chat_bypasses_budget_when_disabled(self) -> None:
        """When enable_reasoning_budget=False."""
        call_count = 0
        received_kwargs: list[dict] = []

        def _chat(*a: Any, **kw: Any) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            received_kwargs.append(dict(kw))
            return _chat_response(content="test")

        self.llm.chat = _chat
        self.llm.allow_reasoning_budget = False
        wrapper = _DecoratedMockLLM(self.llm)

        # Should call underlying method once without budget modifications
        result = wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, max_tokens=1000
        )

        assert call_count == 1
        assert result.message.content == "test"
        # Should pass through original max_tokens, not budget-modified
        assert received_kwargs[0].get("max_tokens") == 1000

    def test_chat_applies_budget_when_enabled(self) -> None:
        """When enable_reasoning_budget=True (default)."""
        call_count = 0

        def _chat(*a: Any, **kw: Any) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call should have budget limit
                assert (
                    kw.get("max_tokens")
                    == _EFFORT_BUDGET_RATIO.get(ReasoningEffort.HIGH) * 1000
                )
                return _truncated_thinking_response()
            else:
                # Second call should have remaining tokens
                return _chat_response(content="final")

        self.llm.chat = _chat
        wrapper = _DecoratedMockLLM(self.llm)
        wrapper.enable_reasoning_budget = True

        result = wrapper.chat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, max_tokens=1000
        )

        assert call_count == 2  # Two-phase call
        assert result.message.content == "final"

    def test_stream_chat_bypasses_budget_when_disabled(self) -> None:
        """Stream chat should bypass budget when flag is disabled."""
        call_count = 0

        def _stream(*a: Any, **kw: Any) -> ChatResponseGen:
            nonlocal call_count
            call_count += 1
            return _sync_gen(_chat_response(content="stream"))

        self.llm.stream_chat = _stream
        wrapper = _DecoratedMockLLM(self.llm)
        wrapper.enable_reasoning_budget = False

        result = list(
            wrapper.stream_chat(self.messages, reasoning_effort=ReasoningEffort.MEDIUM)
        )

        assert call_count == 1
        assert len(result) == 1

    async def test_achat_bypasses_budget_when_disabled(self) -> None:
        """Async chat should bypass budget when flag is disabled."""
        call_count = 0

        async def _achat(*a: Any, **kw: Any) -> ChatResponse:
            nonlocal call_count
            call_count += 1
            return _chat_response(content="async")

        self.llm.achat = _achat
        wrapper = _DecoratedMockLLM(self.llm)
        wrapper.enable_reasoning_budget = False

        result = await wrapper.achat(
            self.messages, reasoning_effort=ReasoningEffort.HIGH, max_tokens=800
        )

        assert call_count == 1
        assert result.message.content == "async"

    async def test_astream_chat_bypasses_budget_when_disabled(self) -> None:
        """Async stream chat should bypass budget when flag is disabled."""
        call_count = 0

        async def _astream(*a: Any, **kw: Any):
            nonlocal call_count
            call_count += 1
            return await _async_gen(_chat_response(content="async_stream"))

        self.llm.astream_chat = _astream
        wrapper = _DecoratedMockLLM(self.llm)
        wrapper.enable_reasoning_budget = False

        gen = await wrapper.astream_chat(
            self.messages, reasoning_effort=ReasoningEffort.LOW
        )
        result = [i async for i in gen]

        assert call_count == 1
        assert len(result) == 1
