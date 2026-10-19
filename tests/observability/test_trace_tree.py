"""A `/v1/messages` turn with a tool call must export exactly one span tree."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import Field, PrivateAttr

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.observability import setup
from private_gpt.observability.processors import (
    DropSpansProcessor,
    FanOutSpanProcessor,
    SemconvSpanProcessor,
)
from private_gpt.server.chat.chat_router import ChatBody
from private_gpt.server.utils.artifact_input import SkillArtifact
from private_gpt.settings.settings import ObservabilitySettings
from tests.server.chat.test_chat_lifecycle_e2e import _create_skill, _skill_tools

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from httpx import AsyncClient
    from llama_index.core.base.llms.types import (
        ChatResponseAsyncGen,
    )

    from tests.fixtures.mock_injector import MockInjector

pytestmark = pytest.mark.anyio


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        DropSpansProcessor(  # ty: ignore[invalid-argument-type]
            patterns=["*.get_metadata", "*._prepare_chat_with_tools"],
            delegate=SemconvSpanProcessor(
                semconv="openinference",
                delegate=FanOutSpanProcessor([SimpleSpanProcessor(memory)]),
            ),
        )
    )
    # The global provider can only be set once per process; route the proxy
    # tracer to this test provider instead.
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    setup._configured = True
    setup._settings = ObservabilitySettings(mode="otel")
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider, use_legacy_callback_handler=False)
    try:
        yield memory
    finally:
        instrumentor.uninstrument()
        setup._reset_for_tests()
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


class _ScriptedLLM(FunctionCallingLLM, ZylonLLM):
    """Real llama-index LLM (so the dispatcher emits spans) with scripted turns."""

    turns: list[list[str | ToolSelection]] = Field(default_factory=list)
    _turn: int = PrivateAttr(default=0)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name="scripted", is_function_calling_model=True, is_chat_model=True
        )

    def get_metadata(self, **kwargs: Any) -> LLMMetadata:
        return self.metadata

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Any = None,
        chat_history: list[ChatMessage] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"messages": list(chat_history or []), "tools": tools, **kwargs}

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> list[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls") or []

    @llm_chat_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        deltas = self.turns[min(self._turn, len(self.turns) - 1)]
        self._turn += 1

        async def gen() -> ChatResponseAsyncGen:
            text = ""
            for delta in deltas:
                tool_calls = [delta] if isinstance(delta, ToolSelection) else None
                text += delta if isinstance(delta, str) else ""
                message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=text or None,
                    additional_kwargs={"tool_calls": tool_calls},
                )
                yield ChatResponse(
                    message=message,
                    delta=delta if isinstance(delta, str) else None,
                    raw={"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                    additional_kwargs=message.additional_kwargs,
                )

        return gen()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        raise NotImplementedError

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError


async def test_messages_turn_with_tool_is_one_trace(
    async_test_client: AsyncClient,
    injector: MockInjector,
    exporter: InMemorySpanExporter,
) -> None:
    collection = str(uuid.uuid4())
    await _create_skill(async_test_client, collection=collection, name="trace-probe")
    exporter.clear()
    llm = _ScriptedLLM(
        message_to_input=Mock(),
        completion_to_input=Mock(),
        turns=[
            [
                ToolSelection(
                    tool_id="t1",
                    tool_name="list_skills",
                    tool_kwargs={"page": 0, "page_size": 20},
                )
            ],
            ["done"],
        ],
    )
    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=llm)
    injector.bind_mock(LLMComponent, llm_component)
    body = ChatBody(
        messages=[MessageInput(content="hello", role="user")],
        tools=_skill_tools(),
        tool_context=[SkillArtifact(skill_filter=SkillFilter(collection=collection))],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200, response.text

    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    turn = by_name["chat.turn"]
    assert turn.attributes["openinference.span.kind"] == "AGENT"

    # Every span of the turn's trace reaches chat.turn through its parents
    # (or is an ancestor of it, i.e. the HTTP server span).
    turn_trace = [s for s in spans if s.context.trace_id == turn.context.trace_id]
    ids = {s.context.span_id: s for s in turn_trace}
    ancestors = set()
    node = turn
    while node.parent is not None and node.parent.span_id in ids:
        node = ids[node.parent.span_id]
        ancestors.add(node.context.span_id)
    for span in turn_trace:
        node = span
        while node.context.span_id != turn.context.span_id and node.parent:
            if node.context.span_id in ancestors or node.parent.span_id not in ids:
                break
            node = ids[node.parent.span_id]
        assert (
            node.context.span_id == turn.context.span_id
            or node.context.span_id in ancestors
        ), f"{span.name} is not under chat.turn"
    # Nothing exported for this request lives in another trace.
    orphans = [
        s.name
        for s in spans
        if s.context.trace_id != turn.context.trace_id
        and not s.name.startswith(("POST /v1/skills", "SkillRepository"))
    ]
    assert not orphans, orphans

    iterations = [s for s in turn_trace if s.name.startswith("chat.iteration")]
    assert len(iterations) == 2
    assert all(s.parent.span_id == turn.context.span_id for s in iterations)

    tool = by_name["tool list_skills"]
    assert tool.attributes["openinference.span.kind"] == "TOOL"
    assert tool.attributes["tool.name"] == "list_skills"
    assert tool.parent.span_id == iterations[0].context.span_id

    llm_spans = [
        s for s in turn_trace if s.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert llm_spans, "LLM spans should nest under the iterations"
    iteration_ids = {s.context.span_id for s in iterations}
    assert all(s.parent.span_id in iteration_ids for s in llm_spans)
    for span in llm_spans:
        for key, value in span.attributes.items():
            if key.endswith("tool.json_schema"):
                assert "object at 0x" not in value

    # Session grouping attributes are copied onto every span of the turn.
    assert {s.attributes.get("session.id") for s in [*iterations, tool]} == {
        turn.attributes["session.id"]
    }
    # Noisy leaves are dropped.
    assert not any(s.name.endswith(".get_metadata") for s in spans)
