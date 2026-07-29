"""Multi-iteration render deduplication through the AsyncChatEngine.

Verifies that across iterations of the async chat loop, when documents
accumulate from one iteration's tool result to the next, the rendered system
prompt delivered to the LLM:

- never contains the same content block twice (no snowballing aggregate, no
  stale duplicates), and
- reflects the *latest* document set — the freshly regenerated
  ``ContextPromptLayer`` survives over any stale aggregate that might embed
  an older version of it.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llm import ToolSelection
from llama_index.core.schema import NodeWithScore, TextNode

from private_gpt.components.chat.models.chat_config_models import (
    CitationConfig,
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.engines.chat.async_chat_engine import (
    AsyncChatCheckpoint,
    AsyncChatEngine,
    IterationCheckpointPayload,
    LocalEventChannel,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatStatus,
)
from private_gpt.components.engines.chat.utils.request_builder import (
    build_request_from_context_stack,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.remote_execution import (
    ToolExecutionResponse,
    build_rebuild_metadata,
    execute_tool_request,
)
from private_gpt.components.tools.tool_scheduler import BaseToolScheduler
from private_gpt.server.chat.interceptors.citation_interceptor import (
    CitationRequestInterceptor,
)
from private_gpt.server.chat.interceptors.document_processing_interceptor import (
    DocumentProcessingRequestInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm


class _FakeAsyncToolScheduler(BaseToolScheduler):
    def __init__(self) -> None:
        self.pending: list[Any] = []
        self._next = 0

    @property
    def is_async(self) -> bool:
        return True

    async def execute(self, request, state_ctx=None, interceptors=None):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def async_execute(self, request, state_ctx=None, interceptors=None) -> str:  # type: ignore[no-untyped-def]
        self._next += 1
        handle = f"handle-{self._next}"
        self.pending.append(request)
        return handle

    async def cancel(self, request, task_id: str | None = None) -> bool:  # type: ignore[no-untyped-def]
        return False

    async def complete_pending(self) -> list[ToolExecutionResponse]:
        responses: list[ToolExecutionResponse] = []
        for request in list(self.pending):
            responses.append(await execute_tool_request(request))
        self.pending.clear()
        return responses


class _FakeChatScheduler:
    async def cancel(self, correlation_id: str) -> bool:
        return True


class _SystemPromptRecorder(ChatRequestLoopInterceptor):
    """Capture the system prompt rendered for the LLM each iteration."""

    def __init__(self) -> None:
        self.prompts_per_iteration: list[Any] = []

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return
        request = build_request_from_context_stack(
            context.state.input.request,
            context.state.input.context_stack,
        )
        self.prompts_per_iteration.append(request.system.prompt)


async def _source_tool(query: str) -> list[NodeWithScore]:
    node = TextNode(
        text="Paris is the capital of France.",
        id_="doc_paris_001",
        metadata={
            "source_id": "src_paris",
            "artifact_id": "art_paris",
            "shorter_id": "ab12",
        },
    )
    return [NodeWithScore(node=node, score=0.95)]


def _rebuild_source_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name, type=name, runtime="server", async_fn=_source_tool
    )


def _server_source_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=name,
        runtime="server",
        async_fn=_source_tool,
        execution_metadata=build_rebuild_metadata(_rebuild_source_tool, {"name": name}),
    )


async def _drain(channel: LocalEventChannel) -> list[Any]:
    return [e async for e in channel.stream()]


def _as_text_list(prompt: Any) -> list[str]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        return [getattr(b, "text", "") for b in prompt]
    return []


def _make_checkpoint(state: Any, resumed_request: Any, responses: Any) -> Any:
    return AsyncChatCheckpoint(
        checkpoint=state.output.pause_type,
        input=ChatInputState(
            request=resumed_request,
            context_stack=state.input.context_stack,
        ),
        iteration=state.runtime.iteration,
        next_block_count=state.runtime.next_block_count,
        payload=IterationCheckpointPayload(
            model_id=state.runtime.model_id,
            pending_async_tools=state.output.pending_async_tools,
            tool_responses=responses,
            pending_external_tool_calls=state.output.pending_external_tool_calls,
            total_input_tokens=state.runtime.total_input_tokens,
            total_output_tokens=state.runtime.total_output_tokens,
            has_input_usage=state.runtime.has_input_usage,
            has_output_usage=state.runtime.has_output_usage,
        ),
    )


def _make_llm_component(mock_llm: Any) -> MagicMock:
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm
    return llm_component


@pytest.mark.asyncio
async def test_latest_documents_survive_across_iterations_no_duplicates() -> None:
    """Two iterations: 0 docs → 1 doc (tool result). Each iteration renders a
    fresh system prompt. Across iterations:

    - Within a single iteration's prompt, no content block may duplicate
      another block (render-time dedup holds per iteration).
    - The second iteration's system prompt must include the freshly
      retrieved document content (latest wins over any stale aggregate).
    """
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="What is Paris?")],
        system=ResolvedSystemConfig(prompt="You are Zylon."),
        citation=CitationConfig(enabled=True),
        tool_config=ResolvedToolConfig(tools=[_server_source_tool("search")]),
    )

    recorder = _SystemPromptRecorder()
    scheduler = _FakeAsyncToolScheduler()
    engine = AsyncChatEngine(
        llm_component=_make_llm_component(
            get_mock_function_calling_llm(
                [
                    [
                        ToolSelection(
                            tool_id="tool_1",
                            tool_name="search",
                            tool_kwargs={"query": "Paris"},
                        )
                    ],
                    ["Paris is the capital of France."],
                ]
            )
        ),
        request_interceptors=[
            CitationRequestInterceptor(),
            DocumentProcessingRequestInterceptor(add_context_to_system_prompt=True),
            recorder,
        ],
        response_interceptors=[],
        max_iterations=6,
        tool_scheduler=scheduler,
        chat_scheduler=_FakeChatScheduler(),
    )

    channel = LocalEventChannel()
    state = await engine.execute(request, channel=channel)
    await channel.close()
    await _drain(channel)

    while state.output.status == ChatStatus.WAITING:
        responses = await scheduler.complete_pending()
        resumed_request = state.input.request.model_copy(deep=True)
        resumed_request.messages = [
            *resumed_request.messages,
            *(response.tool_message for response in responses),
        ]
        channel2 = LocalEventChannel()
        state = await engine.resume(
            _make_checkpoint(state, resumed_request, responses),
            channel=channel2,
        )
        await channel2.close()
        await _drain(channel2)

    assert state.output.status == ChatStatus.COMPLETED
    assert len(recorder.prompts_per_iteration) == 2

    # Per-iteration: no duplicate content block reaches the LLM.
    for n, prompt in enumerate(recorder.prompts_per_iteration, start=1):
        texts = [t.strip() for t in _as_text_list(prompt) if t and t.strip()]
        for i, block in enumerate(texts):
            for j, other in enumerate(texts):
                if i != j and block and block in other:
                    raise AssertionError(
                        f"Iteration {n}: rendered block #{i} is contained in "
                        f"block #{j} — duplicate content reached the LLM.\n"
                        f"blocks: {texts!r}"
                    )

    # Latest documents win: iteration 2 must include the retrieved content
    # (not available on iteration 1).
    first_joined = "\n".join(
        t for t in _as_text_list(recorder.prompts_per_iteration[0]) if t.strip()
    )
    second_joined = "\n".join(
        t for t in _as_text_list(recorder.prompts_per_iteration[1]) if t.strip()
    )
    assert "Paris is the capital of France." not in first_joined, (
        "Document content must NOT appear on iteration 1 (no docs yet): "
        f"{first_joined!r}"
    )
    assert "Paris is the capital of France." in second_joined, (
        "Latest document content must be present in the 2nd iteration prompt: "
        f"{second_joined!r}"
    )
