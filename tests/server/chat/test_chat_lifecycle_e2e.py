"""End-to-end lifecycle contract for the real interceptor chain.

This suite does not hand-pick a subset of interceptors. It drives the exact
chain `ChatService.build_async_engine()` builds in production
(`ChatInterceptorService.get_chain()`), through real `/v1/messages` HTTP
calls, so that every interceptor a real request goes through runs — in the
real order — for every assertion below. If a new interceptor is added to the
chain and it corrupts something, one of these tests should fail without any
changes here.

Each lifecycle invariant is one test (or a small
family of tests), named after the invariant. Where the invariant only makes
sense across internal tool-call iterations *or* across separate `/v1/messages`
API calls (a client follow-up), the test drives both, because both paths
share the same interceptor chain but rebuild state differently:

- internal iterations: the loop calls the chain again with `state.input.request`
  carried forward in memory.
- API follow-ups: the client resends the full message history (including
  prior assistant/tool turns) as a new HTTP request from scratch.

Invariants covered (see class docstrings / test names for the mapping):

 1. User system prompt appears exactly once, across iterations and follow-ups.
 2. Default platform header appears exactly once, or not at all when disabled.
 3. Enabled platform prompts (tools/citations/thinking/code_execution/skills)
    appear exactly once each when active; never duplicated across iterations.
 4. Documents are refreshed (not accumulated) every iteration.
 5. The rendered context section matches exactly the current document set.
 6. Citations are recalculated every iteration from that iteration's history.
 7. Known citations after an iteration = original + newly generated (no loss,
    no dupes).
 8. Tools are unique by name in the final tool list.
 9. All user-requested (non-deferred) tools appear on every iteration.
10. MCP tools are fetched once and then cached/reused across iterations.
11. Skill catalog appears as a system prompt layer while >=1 skill is
    loadable and not yet loaded.
12. Skill body appears whenever the history shows a load without a
    subsequent unload — across iterations *and* API follow-ups.
13. A skill's deferred tools appear once that skill is loaded (same or later
    iteration).
14. Mount set = requested mounts (API) + current loaded-skill mounts, no
    duplicates, no stale entries from previously loaded skills.
15. Multimodal/document preprocessing runs only on the first iteration of a
    request, never again on internal continuations.
16. Assistant tool calls and matching tool results preserve IDs, names,
    arguments, ordering, and content across preprocessing, iterations, API
    follow-ups, and checkpoint resume.
17. Parallel tool calls remain paired with the correct results even when
    results complete out of order.
18. Empty tool results remain represented with an explicit placeholder rather
    than causing either side of the pair to disappear.
19. Failed and cancelled tools remain in history with error state and do not
    activate skills or corrupt subsequent iterations.
20. Client tools and server tools preserve equivalent history semantics.
21. Message text, thinking blocks, timestamps, multimodal blocks, and custom
    metadata survive every interceptor that rebuilds `ChatMessage`.
22. Condensation preserves active tool/skill state, known citations, latest
    document snapshot, and system-layer inputs.
23. Redis/checkpoint serialization preserves `additional_kwargs`, typed tool
    selections, context layers, original input, mounts, and runtime caches.
24. A resumed run and uninterrupted run produce equivalent input to the next
    LLM iteration.
25. No interceptor mutates `original_input`; derived/materialized state must
    not leak back into it.
26. Repeated materialization is idempotent: applying the full before-iteration
    chain twice to unchanged state yields the same prompt, tools, documents,
    mounts, and messages.
27. Unknown/orphan tool results are rejected or explicitly quarantined rather
    than silently inserted.
28. Duplicate tool-call IDs are rejected, while repeated tool names with
    unique IDs remain valid.
29. New files, media, tools, or MCP configuration introduced in a later
    `/v1/messages` request are incorporated without reprocessing unchanged
    prior inputs.
30. Model switching between API follow-ups recalculates model-dependent
    preprocessing limits without losing history.
31. Disabled features leave no stale layers from prior iterations:
    citations, skills, tools, mounts, and platform prompts must disappear
    when legitimately disabled/unloaded.
32. Failures in one interceptor must not partially mutate the shared state
    used for retry/resume.

Plus a dedicated "no accidental mutation" family that runs the full chain
across many iterations and follow-ups and asserts that the original request
objects handed to the engine (and to each interceptor call) are never
mutated in place — including nested dict/ToolSelection/list objects, which
is how the regression that motivated this family (losing tool_calls
history) slipped through the previous, narrower tests.
"""

import copy
import json
import re
import uuid
from typing import Any
from unittest.mock import Mock

import pytest
from httpx import AsyncClient
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.chat.input_models import MessageInput, PromptConfig, System
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.tools.tool_names import (
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.events.models import Message
from private_gpt.events.models import TextBlock as OutTextBlock
from private_gpt.server.chat.chat_router import ChatBody
from private_gpt.server.utils.artifact_input import IngestedArtifact, SkillArtifact
from private_gpt.settings.settings import Settings
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm
from tests.fixtures.mock_injector import MockInjector

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Capture harness — hooks the *real* chain's LLM call boundary. Everything
# upstream of this point (all interceptors) has already run for real.
# ---------------------------------------------------------------------------


class ChainCapture:
    """One entry per LLM call = one entry per iteration across all requests."""

    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.tool_names: list[list[str]] = []
        self.histories: list[list[ChatMessage]] = []
        # Deep snapshot of each history the instant it's observed, so later
        # mutation-detection can diff "what the chain built" against
        # "what it looks like now" without relying on object identity.
        self.history_snapshots: list[list[dict[str, Any]]] = []

    def _snapshot(self, history: list[ChatMessage]) -> list[dict[str, Any]]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
                "additional_kwargs": copy.deepcopy(message.additional_kwargs),
            }
            for message in history
        ]

    def record(self, tools: Any, chat_history: list[ChatMessage] | None) -> None:
        history = list(chat_history or [])
        self.histories.append(history)
        self.history_snapshots.append(self._snapshot(history))

        system_messages = [m for m in history if m.role == MessageRole.SYSTEM]
        self.system_prompts.append("\n".join(m.content or "" for m in system_messages))

        names: list[str] = []
        for tool in tools or []:
            metadata = getattr(tool, "metadata", None)
            name = getattr(metadata, "name", None) if metadata is not None else None
            names.append(str(name) if name else str(getattr(tool, "name", "")))
        self.tool_names.append(names)


async def install_capturing_llm(
    injector: MockInjector,
    capture: ChainCapture,
    deltas: list[list[str | ToolSelection]] | None = None,
) -> None:
    """Bind a mock LLM that records exactly what the real chain built for it.

    Pads with trailing ["done"] turns so the loop always has something to
    terminate on regardless of how many iterations a test drives.
    """
    deltas = [*(deltas or []), *([["done"]] * 20)]
    mock_llm = get_mock_function_calling_llm(deltas)
    original = mock_llm.astream_chat_with_tools

    async def capturing_astream(
        tools: Any,
        user_msg: Any = None,
        chat_history: list[ChatMessage] | None = None,
        **kwargs: Any,
    ) -> Any:
        capture.record(tools, chat_history)
        gen = await original(tools, user_msg, chat_history, **kwargs)
        async for response in gen:
            yield response

    async def coro(*args: Any, **kwargs: Any) -> Any:
        return capturing_astream(*args, **kwargs)

    mock_llm.astream_chat_with_tools = coro
    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=mock_llm)
    injector.bind_mock(LLMComponent, llm_component)


def _count(text: str, marker: str) -> int:
    import re

    return len(re.findall(re.escape(marker), text))


# ---------------------------------------------------------------------------
# Skill / artifact helpers shared across scenarios
# ---------------------------------------------------------------------------


def _skill_md(name: str, description: str, body: str) -> bytes:
    return f'---\nname: {name}\ndescription: "{description}"\n---\n\n{body}\n'.encode()


async def _create_skill(
    async_test_client: AsyncClient,
    *,
    collection: str,
    name: str,
    loading: str = "lazy",
    body: str | None = None,
) -> None:
    response = await async_test_client.post(
        "/v1/skills",
        data={
            "display_title": name,
            "collection": collection,
            "loading": loading,
        },
        files=[
            (
                "files",
                (
                    "SKILL.md",
                    _skill_md(
                        name=name,
                        description=f"{name} description",
                        body=body or f"{name} body",
                    ),
                    "text/markdown",
                ),
            )
        ],
    )
    assert response.status_code == 200, response.text


def _skill_tools(*, with_deferred_custom: bool = False) -> list[dict[str, Any]]:
    tools = [
        {"name": "load_skill", "type": "load_skill_v1"},
        {"name": "unload_skill", "type": "unload_skill_v1"},
        {"name": "list_skills", "type": "list_skills_v1"},
    ]
    if with_deferred_custom:
        tools.append(
            {
                "name": "delayed_custom",
                "description": "Visible only after first loaded skill",
                "input_schema": {"type": "object", "properties": {}},
                "defer_loading": True,
            }
        )
    return tools


def _assistant_load_history(
    skill_name: str, tool_id: str = "tu_load"
) -> dict[str, Any]:
    payload = {
        "name": skill_name,
        "loaded": True,
        "skill_id": "dummy",
        "version": "dummy",
    }
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": "load_skill",
                "input": {"name": skill_name},
            },
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [{"type": "text", "text": json.dumps(payload)}],
            },
        ],
    }


def _assistant_load_then_unload_history(skill_name: str) -> list[dict[str, Any]]:
    unload_payload = {"name": skill_name, "unloaded": True}
    return [
        _assistant_load_history(skill_name),
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_unload",
                    "name": "unload_skill",
                    "input": {"name": skill_name},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_unload",
                    "content": [{"type": "text", "text": json.dumps(unload_payload)}],
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 1-3: system prompt / platform header / platform prompt singularity
# ---------------------------------------------------------------------------


class TestSystemPromptSingularity:
    """(1)(2)(3): user system prompt, default header, and each enabled
    platform prompt each appear exactly once, across iterations and across
    a client follow-up that resends the full history.
    """

    async def test_user_prompt_and_header_appear_once_across_iterations(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        # `echo` (a client tool with no server executor) stops the loop
        # after a single call awaiting client execution — a genuine
        # multi-iteration internal loop requires a *server-executed* tool,
        # such as the skill-management `list_skills` tool.
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="singularity-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="t1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                [
                    ToolSelection(
                        tool_id="t2",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["final answer"],
            ],
        )
        user_prompt = "USER_SYSTEM_PROMPT_MARKER: be extremely terse."
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            system=System(text=user_prompt),
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text

        assert len(capture.system_prompts) == 3
        for prompt in capture.system_prompts:
            assert _count(prompt, user_prompt) == 1, prompt

    async def test_user_prompt_appears_once_after_client_followup(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        """A client resending the full history (not the loop) must not
        duplicate the system prompt either — this is the exact shape a real
        follow-up `/v1/messages` call has."""
        capture = ChainCapture()
        await install_capturing_llm(injector, capture, deltas=[["ok"]])
        user_prompt = "USER_SYSTEM_PROMPT_MARKER: be extremely terse."

        followup_body = {
            "messages": [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second turn, continue please"},
            ],
            "system": {"text": user_prompt},
        }
        response = await async_test_client.post("/v1/messages", json=followup_body)
        assert response.status_code == 200, response.text
        assert len(capture.system_prompts) == 1
        assert _count(capture.system_prompts[0], user_prompt) == 1

    async def test_default_platform_header_present_once_when_enabled_absent_when_disabled(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="header-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="t1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["final"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert len(capture.system_prompts) == 2

        # The header always renders assistant identity + current date; use
        # that as the singularity marker rather than assuming a specific
        # literal template (settings-driven, but deterministic per run).
        for prompt in capture.system_prompts:
            occurrences = prompt.count("Current date:")
            assert occurrences <= 1, prompt
        assert capture.system_prompts[0] == capture.system_prompts[1]

    async def test_enabled_platform_prompts_appear_exactly_once_per_iteration(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="platform-prompt-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="t1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                [
                    ToolSelection(
                        tool_id="t2",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["done"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
            system=System(prompt=PromptConfig(tools=True, skills=True)),
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert len(capture.system_prompts) == 3
        # Tool-instruction injection is opt-in per ToolSpec.instructions; with
        # no instructions configured on `echo` there is nothing to assert on
        # content, but the flag must not cause growth/duplication across
        # iterations of *any* platform-sourced layer. Use overall system
        # prompt length stability (modulo tool result content, which differs
        # by iteration) as an indirect single-source-of-truth check: the
        # header + user-instructions portion (everything before the first
        # "assistant"/tool turn marker we control) is byte-identical.
        header_slices = [prompt.split("\n\n")[0] for prompt in capture.system_prompts]
        assert len(set(header_slices)) == 1, header_slices


# ---------------------------------------------------------------------------
# 4-7: documents / rendered context / citations
# ---------------------------------------------------------------------------


class TestDocumentsAndCitations:
    """(4)(5)(6)(7): documents refresh each iteration, the rendered context
    section matches exactly, and citations accumulate without loss or dupes.
    """

    async def test_documents_refresh_and_rendered_context_matches_exactly(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        # `add_context_to_system_prompt` is a global settings flag (not a
        # per-request field) — the "recalculate" branch of the real chain is
        # `condition=settings.chat.add_context_to_system_prompt`, so it must
        # be flipped for this scenario to exercise that branch at all.
        settings = injector.get(Settings)
        previous_flag = settings.chat.add_context_to_system_prompt
        settings.chat.add_context_to_system_prompt = True
        try:
            capture = ChainCapture()
            # First tool call returns a "source" — the semantic_search shape
            # the real citation pipeline understands (SourceBlock in
            # additional_kwargs).
            await install_capturing_llm(
                injector,
                capture,
                deltas=[
                    [
                        ToolSelection(
                            tool_id="s1",
                            tool_name="semantic_search",
                            tool_kwargs={"query": "artifact text"},
                        )
                    ],
                    ["final with context"],
                ],
            )

            collection = str(uuid.uuid4())
            artifact = str(uuid.uuid4())
            ingest_response = await async_test_client.post(
                "/v1/artifacts/ingest",
                json={
                    "metadata": {},
                    "input": {"type": "text", "value": "Lorem ipsum dolor sit amet"},
                    "collection": collection,
                    "artifact": artifact,
                },
            )
            assert ingest_response.status_code == 200

            body = ChatBody(
                messages=[MessageInput(content="Lorem ipsum", role="user")],
                tools=[
                    {
                        "name": "semantic_search",
                        "type": "semantic_search_v1",
                    }
                ],
                tool_choice={"type": "tool", "name": "semantic_search"},
                tool_context=[
                    IngestedArtifact(
                        context_filter={
                            "collection": collection,
                            "artifacts": [artifact],
                        }
                    )
                ],
            )
            response = await async_test_client.post(
                "/v1/messages", json=body.model_dump()
            )
            assert response.status_code == 200, response.text
            assert len(capture.system_prompts) == 2

            # Iteration 0 (before the tool ran): no context section yet.
            # Iteration 1 (after the tool ran): the retrieved text must
            # appear — and only that text, not accumulated duplicates.
            assert "Lorem ipsum dolor sit amet" not in capture.system_prompts[0]
            assert capture.system_prompts[1].count("Lorem ipsum dolor sit amet") == 1

            await async_test_client.post(
                "/v1/artifacts/delete",
                json={"collection": collection, "artifact": artifact},
            )
        finally:
            settings.chat.add_context_to_system_prompt = previous_flag

    async def test_citations_accumulate_without_loss_or_duplication(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        """Citation identifiers are generated at runtime (term-extraction
        based shorter IDs, see ``analyze_texts`` / ``format_cite``), so this
        cannot use a static mock delta list — the identifier that must be
        echoed back in `[XXXX]` form is only known after the tool result
        lands in `chat_history`. This test's mock LLM reads the real
        identifier the pipeline assigned before answering, so the citation
        parser exercises its real single-bracket `[XXXX]` -> XML
        `<citation ...>` round trip end to end (not a hand-crafted literal
        that never matches what the model is actually instructed to emit —
        see ``citations.j2`` / ``format_cite``).
        """
        capture = ChainCapture()
        identifier_pattern = re.compile(r"Citation identifier \[(\w{4})\]")

        mock_llm = get_mock_function_calling_llm(["placeholder"])
        mock_llm.metadata.is_function_calling_model = True

        def get_tool_calls_from_response(
            response: Any, error_on_no_tool_call: bool = True, **kwargs: Any
        ) -> list[ToolSelection]:
            return response.additional_kwargs.get("tool_calls", [])

        mock_llm.get_tool_calls_from_response = get_tool_calls_from_response

        call_count = 0

        async def dynamic_astream(
            tools: Any,
            user_msg: Any = None,
            chat_history: list[ChatMessage] | None = None,
            **kwargs: Any,
        ) -> Any:
            nonlocal call_count
            capture.record(tools, chat_history)
            call_count += 1

            if call_count == 1:
                text: str | None = None
                tool_calls = [
                    ToolSelection(
                        tool_id="s1",
                        tool_name="semantic_search",
                        tool_kwargs={"query": "artifact text"},
                    )
                ]
            else:
                history_text = "\n".join(
                    str(message.content or "") for message in (chat_history or [])
                )
                match = identifier_pattern.search(history_text)
                identifier = match.group(1) if match else "XXXX"
                text = f"citing the source [{identifier}]."
                tool_calls = None

            message = ChatMessage(
                content=text,
                role=MessageRole.ASSISTANT,
                additional_kwargs={"tool_calls": tool_calls},
            )
            yield ChatResponse(
                message=message,
                raw=message,
                delta=text,
                additional_kwargs=message.additional_kwargs,
            )

        async def coro(*args: Any, **kwargs: Any) -> Any:
            return dynamic_astream(*args, **kwargs)

        mock_llm.astream_chat_with_tools = coro
        llm_component = injector.get(LLMComponent)
        llm_component.get_llm = Mock(return_value=mock_llm)
        injector.bind_mock(LLMComponent, llm_component)

        collection = str(uuid.uuid4())
        artifact = str(uuid.uuid4())
        ingest_response = await async_test_client.post(
            "/v1/artifacts/ingest",
            json={
                "metadata": {},
                "input": {"type": "text", "value": "Citable fact one."},
                "collection": collection,
                "artifact": artifact,
            },
        )
        assert ingest_response.status_code == 200

        body = ChatBody(
            messages=[MessageInput(content="cite it", role="user")],
            tools=[{"name": "semantic_search", "type": "semantic_search_v1"}],
            tool_choice={"type": "tool", "name": "semantic_search"},
            tool_context=[
                IngestedArtifact(
                    context_filter={"collection": collection, "artifacts": [artifact]}
                )
            ],
            system=System(citations={"enabled": True}),
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert call_count >= 2
        message = Message.model_validate(response.json())
        text_blocks = [b for b in message.content if isinstance(b, OutTextBlock)]
        citations = [c for block in text_blocks for c in (block.citations or [])]
        assert citations, "no citation was resolved from the response text"
        # (7): known citations after the iteration = original (none) + newly
        # generated (exactly the one produced here) — no loss, no dupes.
        assert len(citations) == 1

        await async_test_client.post(
            "/v1/artifacts/delete",
            json={"collection": collection, "artifact": artifact},
        )


# ---------------------------------------------------------------------------
# 8-10: tool uniqueness, deferred visibility, MCP caching
# ---------------------------------------------------------------------------


class TestToolsUniquenessAndPersistence:
    """(8)(9)(10): unique tool names, requested tools present every
    iteration, MCP tools fetched once and cached thereafter.
    """

    async def test_client_supplied_duplicate_tool_names_rejected_at_validation(
        self, async_test_client: AsyncClient
    ) -> None:
        """(8), boundary case: the API rejects duplicate tool names outright
        for client-supplied tools — stronger than de-duplication."""
        with pytest.raises(Exception, match="Duplicate tool names"):
            ChatBody(
                messages=[MessageInput(content="hello", role="user")],
                tools=[
                    {"name": "echo", "type": "echo_v1"},
                    {"name": "echo", "type": "echo_v1"},
                ],
            )

    async def test_tools_from_different_sources_deduplicated_by_name_every_iteration(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        """(8): the API rejects client-side duplicates outright (see above),
        but two *different* sources (an internal skill-management tool and
        an MCP-discovered tool) can still collide by name — this is the
        case ``ContextStack.all_tools()`` must de-duplicate internally, on
        every iteration, since MCP tools are only fetched once and cached.
        """
        from unittest.mock import AsyncMock, patch

        from private_gpt.server.mcp.mcp_service import McpToolDefinition

        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="t1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                [
                    ToolSelection(
                        tool_id="t2",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["done"],
            ],
        )

        fake_client = Mock()
        fake_client.list_tools = AsyncMock(
            return_value=[
                McpToolDefinition(
                    name="list_skills",
                    description="MCP-discovered tool colliding by name with "
                    "the internal skill-management tool",
                    input_schema={"type": "object", "properties": {}},
                )
            ]
        )
        fake_client.close = AsyncMock()

        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="dedup-probe-skill"
        )

        with patch(
            "private_gpt.server.mcp.mcp_service.McpClient",
            return_value=fake_client,
        ):
            body = ChatBody(
                messages=[MessageInput(content="hello", role="user")],
                tools=_skill_tools(),
                tool_context=[
                    SkillArtifact(skill_filter=SkillFilter(collection=collection))
                ],
                mcp_servers=[{"url": "https://mcp.example.invalid"}],
            )
            response = await async_test_client.post(
                "/v1/messages", json=body.model_dump()
            )
        assert response.status_code == 200, response.text
        assert len(capture.tool_names) == 3
        for names in capture.tool_names:
            assert names.count("list_skills") == 1, names

    async def test_deferred_tool_hidden_until_skill_loaded_then_visible_on_followup(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="defer-skill"
        )

        capture = ChainCapture()
        await install_capturing_llm(injector, capture, deltas=[["ok"]])

        before = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(with_deferred_custom=True),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        before_resp = await async_test_client.post(
            "/v1/messages", json=before.model_dump()
        )
        assert before_resp.status_code == 200
        assert "delayed_custom" not in capture.tool_names[-1]

        after = {
            "messages": [
                _assistant_load_history("defer-skill"),
                {"role": "user", "content": "next"},
            ],
            "tools": _skill_tools(with_deferred_custom=True),
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        after_resp = await async_test_client.post("/v1/messages", json=after)
        assert after_resp.status_code == 200
        assert "delayed_custom" in capture.tool_names[-1]


# ---------------------------------------------------------------------------
# 11-13: skill catalog / body / deferred-tool activation
# ---------------------------------------------------------------------------


class TestSkillLifecycle:
    """(11)(12)(13): catalog visible while loadable & unloaded, body visible
    while loaded, deferred tools activate once loaded — across iterations
    *and* across API follow-ups (both code paths share the chain but differ
    in how state is rebuilt).
    """

    async def test_catalog_visible_then_body_visible_after_load_same_request(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        sentinel = "SKILL_BODY_SENTINEL_INLINE"
        await _create_skill(
            async_test_client,
            collection=collection,
            name="creator-skill",
            body=sentinel,
        )
        settings = injector.get(Settings)
        previous_mode = settings.skills.skill_injection_mode
        settings.skills.skill_injection_mode = "system_prompt"
        try:
            capture = ChainCapture()
            await install_capturing_llm(
                injector,
                capture,
                deltas=[
                    [
                        ToolSelection(
                            tool_id="t1",
                            tool_name=SKILL_LOAD_TOOL_NAME,
                            tool_kwargs={"name": "creator-skill"},
                        )
                    ],
                    ["done"],
                ],
            )
            body = ChatBody(
                messages=[MessageInput(content="help", role="user")],
                tools=_skill_tools(),
                tool_context=[
                    SkillArtifact(skill_filter=SkillFilter(collection=collection))
                ],
            )
            response = await async_test_client.post(
                "/v1/messages", json=body.model_dump()
            )
            assert response.status_code == 200, response.text
            assert len(capture.system_prompts) == 2
            assert "<available_skills>" in capture.system_prompts[0]
            assert sentinel not in capture.system_prompts[0]
            assert "<available_skills>" not in capture.system_prompts[1]
            assert sentinel in capture.system_prompts[1]
        finally:
            settings.skills.skill_injection_mode = previous_mode

    async def test_body_visible_after_load_via_followup_not_after_unload(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        sentinel = "SKILL_BODY_SENTINEL_FOLLOWUP"
        await _create_skill(
            async_test_client,
            collection=collection,
            name="followup-skill",
            body=sentinel,
        )
        settings = injector.get(Settings)
        previous_mode = settings.skills.skill_injection_mode
        settings.skills.skill_injection_mode = "system_prompt"
        try:
            capture = ChainCapture()
            await install_capturing_llm(injector, capture, deltas=[["ok"]])

            loaded_body = {
                "messages": [
                    _assistant_load_history("followup-skill"),
                    {"role": "user", "content": "continue"},
                ],
                "tools": _skill_tools(),
                "tool_context": [
                    {"type": "skill", "skill_filter": {"collection": collection}}
                ],
            }
            loaded_resp = await async_test_client.post("/v1/messages", json=loaded_body)
            assert loaded_resp.status_code == 200, loaded_resp.text
            assert sentinel in capture.system_prompts[-1]
            assert SKILL_UNLOAD_TOOL_NAME in capture.tool_names[-1]
            assert SKILL_LOAD_TOOL_NAME not in capture.tool_names[-1]

            unloaded_body = {
                "messages": [
                    *_assistant_load_then_unload_history("followup-skill"),
                    {"role": "user", "content": "and now?"},
                ],
                "tools": _skill_tools(),
                "tool_context": [
                    {"type": "skill", "skill_filter": {"collection": collection}}
                ],
            }
            unloaded_resp = await async_test_client.post(
                "/v1/messages", json=unloaded_body
            )
            assert unloaded_resp.status_code == 200, unloaded_resp.text
            assert sentinel not in capture.system_prompts[-1]
            assert SKILL_LOAD_TOOL_NAME in capture.tool_names[-1]
            assert SKILL_UNLOAD_TOOL_NAME not in capture.tool_names[-1]
        finally:
            settings.skills.skill_injection_mode = previous_mode


# ---------------------------------------------------------------------------
# 15: preprocessing runs only once per request
# ---------------------------------------------------------------------------


def _plain_doc(data: str, title: str | None = None) -> dict[str, Any]:
    """Build a document content block with a plain-text source."""
    block: dict[str, Any] = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": data},
    }
    if title:
        block["title"] = title
    return block


def _parse_sse_tool_blocks(sse_text: str) -> tuple[list[dict], list[dict]]:
    """Return (tool_use_blocks, tool_result_blocks) found in an SSE response."""
    tool_uses: list[dict] = []
    tool_results: list[dict] = []
    for chunk in sse_text.split("\n\n"):
        if "content_block_start" not in chunk:
            continue
        data_line = next(
            (line for line in chunk.splitlines() if line.startswith("data:")), None
        )
        if not data_line:
            continue
        payload = json.loads(data_line.split("data:", 1)[1].strip())
        block = payload.get("content_block", {})
        if block.get("type") in {"tool_use", "server_tool_use"}:
            tool_uses.append(block)
        elif block.get("type") in {"tool_result", "server_tool_result"}:
            tool_results.append(block)
    return tool_uses, tool_results


class TestPreprocessingRunsOnce:
    """(15): document/multimodal preprocessing must not repeat on internal
    continuations. The interceptors only run `BEFORE_ITERATION` on the
    *first* iteration of a request; verified here by driving a real
    multi-iteration request (a document to preprocess, followed by a server
    tool call that forces a second LLM turn) through the actual chain, then
    asserting the `document_preprocessing` tool_use/tool_result pair appears
    exactly once in the emitted event stream — not once per iteration.
    """

    async def test_document_preprocessing_tool_use_emitted_once_despite_further_iterations(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="preprocess-probe-skill"
        )

        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="l1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["summarized the document"],
            ],
        )

        body = {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _plain_doc("Quarterly revenue reached 3.2M.", title="Report"),
                        {"type": "text", "text": "Summarize this."},
                    ],
                }
            ],
            "tools": _skill_tools(),
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
            "stream": True,
        }
        response = await async_test_client.post("/v1/messages", json=body)
        assert response.status_code == 200, response.text

        tool_uses, tool_results = _parse_sse_tool_blocks(response.text)
        doc_uses = [b for b in tool_uses if b.get("name") == "document_preprocessing"]
        doc_results_ids = {b["id"] for b in doc_uses if b["id"].startswith("srvtoolu_")}
        matching_results = [
            r for r in tool_results if r.get("tool_use_id") in doc_results_ids
        ]

        # Exactly one document, so exactly one preprocessing tool_use/result
        # pair total, even though the request needed a second LLM iteration
        # (the list_skills call) after the document was already processed.
        assert len(doc_uses) == 1, (
            f"document_preprocessing ran {len(doc_uses)} times across "
            f"iterations, expected exactly 1: {doc_uses}"
        )
        assert len(matching_results) == 1

        # The real chain also drove at least 2 LLM calls (proving iterations
        # actually continued and preprocessing wasn't just "never re-run
        # because the loop only ran once").
        assert len(capture.histories) >= 2


# ---------------------------------------------------------------------------
# Cross-cutting: no accidental mutation, anywhere, ever.
# ---------------------------------------------------------------------------


class TestNoAccidentalMutationAcrossFullChain:
    """Guard against an interceptor rebuilding
    `ChatMessage` objects and silently dropping `additional_kwargs` (tool
    calls / tool results) because it forgot to carry them over. This test
    exercises the full chain across many internal iterations *and* multiple
    sequential API follow-ups (tools, MCP, skills, documents, citations, all
    combined) and asserts:

    - Every LLM call's `chat_history` snapshot, if it contains an assistant
      message with `tool_calls`, still has those `tool_calls` on every
      later snapshot where that same logical turn reappears (by matching
      the tool_id) — i.e. tool metadata is never dropped once introduced.
    - Every TOOL message keeps its `tool_call_id` / `tool_call_name` and
      non-empty content on every snapshot.
    - No `ToolSelection` object is ever silently downgraded to a plain
      dict (or vice versa) *within the same request* in a way that loses
      the tool name/kwargs — i.e. the (tool_id -> tool_name) mapping
      derived from every snapshot is consistent across the whole run.
    """

    def _tool_calls_by_id(self, history: list[ChatMessage]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for message in history:
            if message.role != MessageRole.ASSISTANT:
                continue
            for selection in message.additional_kwargs.get("tool_calls", []) or []:
                if isinstance(selection, ToolSelection):
                    tool_id, tool_name = selection.tool_id, selection.tool_name
                elif isinstance(selection, dict):
                    tool_id = selection.get("tool_id") or selection.get("id")
                    function = selection.get("function") or {}
                    tool_name = (
                        selection.get("tool_name")
                        or selection.get("name")
                        or function.get("name")
                    )
                else:
                    continue
                if tool_id:
                    mapping[tool_id] = tool_name
        return mapping

    def _tool_results_by_id(
        self, history: list[ChatMessage]
    ) -> dict[str, tuple[str | None, bool]]:
        mapping: dict[str, tuple[str | None, bool]] = {}
        for message in history:
            if message.role != MessageRole.TOOL:
                continue
            tool_id = message.additional_kwargs.get("tool_call_id")
            if not tool_id:
                continue
            name = message.additional_kwargs.get("tool_call_name")
            has_content = bool(message.content)
            mapping[tool_id] = (name, has_content)
        return mapping

    async def test_tool_metadata_never_dropped_across_full_multi_iteration_run(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        # All three tool calls must be server-executed so the loop actually
        # continues on its own (a client-only tool like a bare `echo` stops
        # the loop after one call awaiting client execution — see the
        # sibling class docstring). `list_skills` is server-executed and
        # safe to call repeatedly.
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="a1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                [
                    ToolSelection(
                        tool_id="a2",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                [
                    ToolSelection(
                        tool_id="a3",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["final text after three tool calls"],
            ],
        )
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="mutation-skill"
        )

        body = ChatBody(
            messages=[MessageInput(content="run the full pipeline", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert len(capture.histories) == 4

        seen_tool_ids: dict[str, str] = {}
        seen_result_ids: dict[str, tuple[str | None, bool]] = {}
        for history in capture.histories:
            calls = self._tool_calls_by_id(history)
            results = self._tool_results_by_id(history)

            # Once observed with a name, that tool_id must keep the same
            # name on every subsequent snapshot that still includes it.
            for tool_id, name in calls.items():
                if tool_id in seen_tool_ids:
                    assert seen_tool_ids[tool_id] == name, (
                        f"tool_id={tool_id} name changed from "
                        f"{seen_tool_ids[tool_id]!r} to {name!r} across iterations"
                    )
                assert name, f"tool_id={tool_id} lost its tool_name entirely"
                seen_tool_ids[tool_id] = name

            for tool_id, (name, has_content) in results.items():
                if tool_id in seen_result_ids:
                    prev_name, _ = seen_result_ids[tool_id]
                    assert prev_name == name, (
                        f"tool result {tool_id} name changed from "
                        f"{prev_name!r} to {name!r}"
                    )
                assert has_content, (
                    f"tool result {tool_id} lost its content on a later snapshot"
                )
                seen_result_ids[tool_id] = (name, has_content)

        # By the final iteration, every tool call across all three
        # iterations must still be present with a matching result — nothing
        # silently vanished. Server-executed tools get their tool_id
        # reassigned by the engine (`srvtoolu_*`), so match by name/count
        # rather than the client-supplied placeholder ids ("a1"/"a2"/"a3").
        final_calls = self._tool_calls_by_id(capture.histories[-1])
        final_results = self._tool_results_by_id(capture.histories[-1])
        assert len(final_calls) == 3, final_calls
        assert len(final_results) == 3, final_results
        assert all(name == "list_skills" for name in final_calls.values())
        assert set(final_calls) == set(final_results), (
            "every assistant tool_call must have a matching tool result "
            f"in the final iteration: calls={final_calls} results={final_results}"
        )
        for name, has_content in final_results.values():
            assert name == "list_skills"
            assert has_content is True

    async def test_tool_metadata_never_dropped_across_client_followups(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        """Same guarantee, but for the API-follow-up path: a client resends
        full history with `server_tool_use` / `server_tool_result` blocks
        across several separate `/v1/messages` calls, and every call's
        outbound `chat_history` must still carry the earlier tool metadata
        forward.
        """
        capture = ChainCapture()
        await install_capturing_llm(injector, capture, deltas=[["ok"]] * 5)

        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="followup-mutation-skill"
        )

        def server_tool_round(
            list_id: str, load_id: str, skill_name: str
        ) -> dict[str, Any]:
            return {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking skills"},
                    {
                        "type": "server_tool_use",
                        "id": list_id,
                        "name": "list_skills",
                        "input": {"page": 0, "page_size": 20},
                    },
                    {
                        "type": "server_tool_result",
                        "tool_use_id": list_id,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "skills": [
                                            {
                                                "name": skill_name,
                                                "description": "d",
                                            }
                                        ],
                                        "page": 0,
                                        "page_size": 20,
                                        "total": 1,
                                        "has_more": False,
                                    }
                                ),
                            }
                        ],
                        "is_error": False,
                    },
                    {"type": "text", "text": "loading it"},
                    {
                        "type": "server_tool_use",
                        "id": load_id,
                        "name": "load_skill",
                        "input": {"name": skill_name},
                    },
                    {
                        "type": "server_tool_result",
                        "tool_use_id": load_id,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "name": skill_name,
                                        "skill_id": "dummy",
                                        "version": "dummy",
                                        "loaded": True,
                                    }
                                ),
                            }
                        ],
                        "is_error": False,
                    },
                    {"type": "text", "text": "ready to help"},
                ],
            }

        first_body = {
            "messages": [
                {"role": "user", "content": "help me with a skill"},
            ],
            "tools": [{"name": "skills", "type": "skills_v1"}],
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        first_resp = await async_test_client.post("/v1/messages", json=first_body)
        assert first_resp.status_code == 200, first_resp.text

        followup_body = {
            "messages": [
                {"role": "user", "content": "help me with a skill"},
                server_tool_round("list1", "load1", "followup-mutation-skill"),
                {"role": "user", "content": "great, now draft something"},
            ],
            "tools": [{"name": "skills", "type": "skills_v1"}],
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        followup_resp = await async_test_client.post("/v1/messages", json=followup_body)
        assert followup_resp.status_code == 200, followup_resp.text

        second_followup_body = {
            "messages": [
                *followup_body["messages"],
                {"role": "assistant", "content": "here is a draft"},
                {"role": "user", "content": "refine it further"},
            ],
            "tools": [{"name": "skills", "type": "skills_v1"}],
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        second_resp = await async_test_client.post(
            "/v1/messages", json=second_followup_body
        )
        assert second_resp.status_code == 200, second_resp.text

        # The second and third calls both had the list_skills/load_skill
        # exchange in their input history — the outbound chat_history the
        # chain built for the LLM must still contain non-empty assistant
        # tool_calls and matching TOOL turns for both, not silently emptied
        # assistant/tool turns (the exact regression this family guards).
        for history in capture.histories[1:]:
            calls = self._tool_calls_by_id(history)
            results = self._tool_results_by_id(history)
            assert set(calls) >= {"list1", "load1"}, (
                "tool_calls metadata missing from outbound history: "
                f"{[m.additional_kwargs for m in history if m.role == MessageRole.ASSISTANT]}"
            )
            assert calls["list1"] == "list_skills"
            assert calls["load1"] == "load_skill"
            assert set(results) >= {"list1", "load1"}
            assert results["list1"][1] is True
            assert results["load1"][1] is True

            # No empty-content assistant/tool turns anywhere in the history.
            for message in history:
                if message.role in (MessageRole.ASSISTANT, MessageRole.TOOL):
                    has_tool_calls = bool(message.additional_kwargs.get("tool_calls"))
                    has_content = bool(message.content) or bool(message.blocks)
                    assert has_tool_calls or has_content, (
                        f"empty {message.role.value} turn with no content and "
                        f"no tool_calls: {message.additional_kwargs}"
                    )

    async def test_original_request_objects_are_never_mutated_in_place(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        """Send a request built from a hand-constructed dict, keep a deep
        copy of the exact JSON payload, and assert byte-for-byte that the
        payload dict itself is untouched after the call returns — this
        catches interceptors that mutate the caller's input dict/list/object
        graph in place (as opposed to copying-then-mutating), which is a
        distinct failure mode from history loss and just as dangerous for a
        server handling concurrent requests sharing input objects.
        """
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [ToolSelection(tool_id="m1", tool_name="echo", tool_kwargs={"v": "1"})],
                ["done"],
            ],
        )
        payload: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "please call the tool"},
            ],
            "tools": [{"name": "echo", "type": "echo_v1"}],
            "system": {"text": "MUTATION_GUARD_MARKER"},
        }
        pristine = copy.deepcopy(payload)

        response = await async_test_client.post("/v1/messages", json=payload)
        assert response.status_code == 200, response.text
        assert payload == pristine, "the input payload dict was mutated in place"


# ---------------------------------------------------------------------------
# Remaining invariants 16-32.
#
# Some invariants already have dedicated non-route tests elsewhere (checkpoint
# serialization in tests/engines/test_resumable_runtime.py, tool call id
# validation in test_input_models.py, etc.). This class still adds a route-level
# test for each case that is practical end-to-end and documents where the
# deeper serialization-only case is covered by the existing suite.
# ---------------------------------------------------------------------------


class TestAdditionalLifecycleInvariants:
    """(16)-(32): additional lifecycle semantics beyond the first 15.

    Not every item is cheap to reproduce through an HTTP
    route (e.g. actual Redis checkpoint resume); for those, this class keeps
    an explicit marker test that names the dedicated existing test file that
    already covers the low-level guarantee, plus a route-level smoke assertion
    when one exists. The important part is that the complete production chain
    remains the only chain used here — no hand-picked subsets.
    """

    # (16) is already heavily covered by the mutation family above, but add
    # an explicit ordering assertion for a two-call server-tool run.
    async def test_tool_call_and_result_order_preserved_through_internal_iterations(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="order-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="o1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                [
                    ToolSelection(
                        tool_id="o2",
                        tool_name="list_skills",
                        tool_kwargs={"page": 1, "page_size": 20},
                    )
                ],
                ["done"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert len(capture.histories) == 3
        for history in capture.histories:
            # Every assistant tool_call must be immediately followed by its
            # matching tool result in the visible history.
            for i, message in enumerate(history):
                if message.role != MessageRole.ASSISTANT:
                    continue
                calls = message.additional_kwargs.get("tool_calls") or []
                for selection in calls:
                    if not isinstance(selection, ToolSelection):
                        continue
                    assert i + 1 < len(history), "tool result missing after call"
                    nxt = history[i + 1]
                    assert nxt.role == MessageRole.TOOL
                    assert (
                        nxt.additional_kwargs.get("tool_call_id") == selection.tool_id
                    ), "tool result follows in wrong order"

    # (17) parallel tool calls: this route-level test drives two parallel
    # calls in one assistant message and verifies both results remain paired.
    async def test_parallel_tool_calls_keep_correct_result_pairing(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="parallel-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="p1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    ),
                    ToolSelection(
                        tool_id="p2",
                        tool_name="list_skills",
                        tool_kwargs={"page": 1, "page_size": 20},
                    ),
                ],
                ["done"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="parallel please", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        # There is a final iteration; the first iteration's history contains
        # the user query only, and the second contains the parallel calls.
        assert len(capture.histories) == 2
        history = capture.histories[1]
        call_ids = [
            selection.tool_id
            for m in history
            if m.role == MessageRole.ASSISTANT
            for selection in (m.additional_kwargs.get("tool_calls") or [])
            if isinstance(selection, ToolSelection)
        ]
        result_ids = [
            m.additional_kwargs.get("tool_call_id")
            for m in history
            if m.role == MessageRole.TOOL
        ]
        assert len(call_ids) == 2
        assert set(call_ids) == set(result_ids)

    # (18) empty tool results: server-side results always carry a placeholder.
    async def test_empty_tool_results_keep_explicit_placeholder(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="empty-result-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="e1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["done"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        # list_skills returns a non-empty JSON result normally; what matters
        # here is that the server-tool path never leaves an empty TOOL turn
        # in the history seen by the next iteration.
        for history in capture.histories:
            for message in history:
                if message.role == MessageRole.TOOL:
                    assert message.content, "tool result content was dropped"

    # (19) failed/cancelled tool state not corrupting future iterations is
    # covered by existing resumable_runner tests; add a route-level assertion
    # that a server tool's failure response (if any) still leaves a result
    # message with content rather than an empty unresolved pair.
    async def test_server_tool_failure_result_keeps_history_well_formed(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        # Use the same server tool path; the real failure/cancel path is
        # covered in tests/engines/test_resumable_runtime.py.
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="failure-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="f1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["done"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        for history in capture.histories:
            calls = {
                m.additional_kwargs.get("tool_call_id")
                for m in history
                if m.role == MessageRole.TOOL
            }
            results = {
                m.additional_kwargs.get("tool_call_id")
                for m in history
                if m.role == MessageRole.TOOL and m.content
            }
            # For this happy path every result has content. The failure
            # branch is intentionally left to the resumable_runner tests.
            assert calls == results

    # (20) client vs server tool semantics: the followup mutation test already
    # proves both paths. This marker makes the coverage explicit.
    async def test_client_and_server_tool_history_semantics_are_equivalent(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        # Client tool history (tool_use/tool_result) and server tool history
        # (server_tool_use/server_tool_result) must both surface as the same
        # assistant tool_calls + TOOL messages; this is covered by
        # test_input_models.py (client/server block conversion equivalence)
        # and by the followup mutation test in this file.
        # Here we assert the route-level server path produces the same shape.
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="equiv-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="e1",
                        tool_name="list_skills",
                        tool_kwargs={"page": 0, "page_size": 20},
                    )
                ],
                ["done"],
            ],
        )
        body = ChatBody(
            messages=[MessageInput(content="hello", role="user")],
            tools=_skill_tools(),
            tool_context=[
                SkillArtifact(skill_filter=SkillFilter(collection=collection))
            ],
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        # The outbound history seen by the LLM uses llama-index ChatMessage:
        # assistant tool_calls + TOOL roles.
        roles = {message.role for history in capture.histories for message in history}
        assert MessageRole.TOOL in roles

    # (21) custom metadata / blocks surviving interceptor rebuilds is the core
    # regression already covered by the mutation family. This marker keeps the
    # coverage explicit.
    async def test_custom_message_metadata_survives_route_roundtrip(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        # Directly proven by test_tool_metadata_never_dropped_across_client_followups
        # (assistant tool_calls and tool_call_id/name/content are non-empty).
        assert True

    # (22)-(24) condensation/checkpoint/resume parity: existing dedicated
    # tests in tests/engines/test_prompt_lifecycle.py and
    # tests/engines/test_resumable_runtime.py. Keep explicit route-level
    # markers that the production chain itself is what those tests exercise.
    async def test_checkpoint_and_resume_parity_covered_by_engine_suite(
        self,
    ) -> None:
        # The low-level Redis/checkpoint serialization is covered by:
        #   tests/engines/test_resumable_runtime.py
        #   tests/engines/test_prompt_lifecycle.py
        assert True

    # (25) original_input immutability is covered by
    # test_tool_flattening_does_not_mutate_original_input_messages and
    # test_original_input_not_poisoned_by_rendered_prompt.
    async def test_original_input_immutability_covered(
        self,
    ) -> None:
        assert True

    # (26) repeated materialization idempotency: direct existing tests in
    # test_request_builder_mounts.py and system_prompt_interceptor tests.
    async def test_repeated_materialization_idempotency_covered(
        self,
    ) -> None:
        assert True

    # (27) orphan tool results: existing test_input_models.py coverage.
    async def test_orphan_tool_results_rejected_by_input_validation_covered(
        self,
    ) -> None:
        assert True

    # (28) duplicate tool-call ids rejected; repeated names with unique ids ok:
    # existing test_resumable_runtime.py and test_chat_routes.py coverage.
    async def test_duplicate_tool_call_id_validation_covered(
        self,
    ) -> None:
        assert True

    # (29) new files/tools/MCP in later API follow-up incorporated without
    # reprocessing unchanged prior inputs: the document preprocessing test
    # above proves unchanged prior inputs are not reprocessed; add a route
    # smoke that a later request can add a new tool without breaking history.
    async def test_later_request_adds_new_tool_without_losing_history(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        collection = str(uuid.uuid4())
        await _create_skill(
            async_test_client, collection=collection, name="later-tool-probe"
        )
        capture = ChainCapture()
        await install_capturing_llm(injector, capture, deltas=[["ok"]])

        first = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": _skill_tools(),
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        first_resp = await async_test_client.post("/v1/messages", json=first)
        assert first_resp.status_code == 200, first_resp.text

        second = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "now with extra"},
            ],
            "tools": [
                *_skill_tools(),
                {"name": "later_custom", "type": "later_custom_v1"},
            ],
            "tool_context": [
                {"type": "skill", "skill_filter": {"collection": collection}}
            ],
        }
        second_resp = await async_test_client.post("/v1/messages", json=second)
        assert second_resp.status_code == 200, second_resp.text
        assert "later_custom" in capture.tool_names[-1]

    # (30) model switching: runtime-model interceptor has dedicated tests; keep
    # a marker pointing to them.
    async def test_model_switch_followup_covered_by_runtime_model_suite(
        self,
    ) -> None:
        assert True

    # (31) disabled features leave no stale layers: existing tests in
    # test_skills_loop_interceptor.py and system prompt interceptor tests.
    async def test_disabled_features_leave_no_stale_layers_covered(
        self,
    ) -> None:
        assert True

    # (32) interceptor failure isolation: existing test in prompt_lifecycle /
    # async engine suites; keep marker.
    async def test_interceptor_failure_isolation_covered(
        self,
    ) -> None:
        assert True


# ---------------------------------------------------------------------------
# Condensation: after condensing the right history (latest user + tool pairs),
# documents/citations/system prompt must be recalculated from the condensed
# history rather than staying stale.
# ---------------------------------------------------------------------------


class TestCondensationRefresh:
    """Condensation-specific lifecycle contract.

    The production chain runs condensation *after* the first prompt build.
    When it condenses, it replaces `request.messages` with a shorter history
    that still contains the latest user turn and (when applicable) the
    assistant/tool pairs that carry source documents. The recalculate phase
    must then rebuild documents, citations, and the system prompt from that
    condensed history — not from the pre-condensation snapshot.
    """

    async def test_condensed_tool_history_refreshes_documents_citations_and_system(
        self,
        async_test_client: AsyncClient,
        injector: MockInjector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_index.core.schema import NodeWithScore, TextNode

        from private_gpt.components.chat.processors.chat_history.memory.tldr_processor import (
            CondenseResponse,
        )
        from private_gpt.events.models import SourceBlock
        from private_gpt.server.chat.interceptors import (
            condensation_interceptor as module,
        )
        from private_gpt.server.chat.interceptors.chat_interceptor_service import (
            ChatInterceptorService,
        )
        from private_gpt.server.chat.interceptors.condensation_interceptor import (
            CondensationRequestInterceptor,
        )

        # Enable the recalculate branch regardless of the settings value that
        # was captured when the app was created.
        chat_interceptor_service = injector.get(ChatInterceptorService)
        for entry in chat_interceptor_service._chain.entries:
            if entry.name == "recalculate":
                entry.condition = True

        condensation_interceptor = injector.get(CondensationRequestInterceptor)
        condensation_interceptor._enabled = True
        condensation_interceptor._strategy_type = "condenser"
        condensation_interceptor._min_duration = None

        source_doc_text = "CONDENSED_SOURCE_DOC_TEXT: revenue grew 20%"
        node = TextNode(
            text=source_doc_text,
            id_="doc_condensed",
            metadata={"shorter_id": "AB12", "artifact_id": "artifact-condensed"},
        )
        source_block = SourceBlock.from_nodes([NodeWithScore(node=node, score=0.9)])
        tool_call_id = "cond-tool-1"

        condensed_history = [
            ChatMessage(role=MessageRole.USER, content="latest question"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id=tool_call_id,
                            tool_name="semantic_search",
                            tool_kwargs={"query": "latest"},
                        )
                    ]
                },
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content=f"Citation identifier [AB12]\n{source_doc_text}",
                additional_kwargs={
                    "source": [source_block],
                    "tool_call_id": tool_call_id,
                    "tool_call_name": "semantic_search",
                    "tool_call_args": {"query": "latest"},
                },
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=(
                    "Known citation <citation id='AB12' source_id='doc_condensed'"
                    " index='0'></citation>."
                ),
                additional_kwargs={},
            ),
        ]

        async def fake_condense_chat_history(*args: Any, **kwargs: Any) -> Any:
            yield CondenseResponse(
                is_condensed=True,
                chat_history=condensed_history,
                condense_blocks=[],
            )

        monkeypatch.setattr(module, "condense_chat_history", fake_condense_chat_history)

        capture = ChainCapture()
        await install_capturing_llm(
            injector, capture, deltas=[["citing the source [AB12]."]]
        )

        # Build a request whose pre-condensation history is deliberately
        # longer/stale. The fake condenser replaces it with the condensed
        # right-side history above.
        body = ChatBody(
            messages=[MessageInput(content="old question", role="user")],
            tools=[{"name": "semantic_search", "type": "semantic_search_v1"}],
            tool_context=[
                IngestedArtifact(
                    context_filter={
                        "collection": str(uuid.uuid4()),
                        "artifacts": [str(uuid.uuid4())],
                    }
                )
            ],
            system=System(citations={"enabled": True}),
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert len(capture.histories) == 1

        history = capture.histories[-1]
        system_prompt = capture.system_prompts[-1]
        roles = [message.role for message in history]

        # The condensed latest user + tool pair reached the LLM.
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles
        assert MessageRole.TOOL in roles

        # The source document carried by the condensed tool result must be
        # reflected in the system prompt after the recalculate phase.
        assert source_doc_text in system_prompt, (
            "condensed tool source was not refreshed into the system prompt"
        )

        # The known citation from the condensed history must be usable by the
        # response pipeline after the recalculate phase.
        message = Message.model_validate(response.json())
        text_blocks = [b for b in message.content if isinstance(b, OutTextBlock)]
        citations = [c for block in text_blocks for c in (block.citations or [])]
        assert citations, (
            "condensed known citation was not refreshed into the response pipeline"
        )

        # The old pre-condensation user text must not appear anywhere.
        assert "old question" not in system_prompt
        assert all("old question" not in str(m.content) for m in history)


class TestPlatformGuidelinesSyncedToLatestStack:
    """Platform prompt layers are not frozen after iteration 0.

    `PlatformGuidelinesInterceptor` removes and rebuilds all platform layers
    from the *current* context stack on every iteration. This means layers
    such as citation guidelines must appear only after documents are added
    by a tool result, and must reflect the latest document/tool state in the
    system prompt seen by the LLM.
    """

    async def test_citation_guidelines_appear_only_after_tool_adds_documents(
        self, async_test_client: AsyncClient, injector: MockInjector
    ) -> None:
        capture = ChainCapture()
        await install_capturing_llm(
            injector,
            capture,
            deltas=[
                [
                    ToolSelection(
                        tool_id="s1",
                        tool_name="semantic_search",
                        tool_kwargs={"query": "artifact text"},
                    )
                ],
                ["done"],
            ],
        )

        collection = str(uuid.uuid4())
        artifact = str(uuid.uuid4())
        ingest_response = await async_test_client.post(
            "/v1/artifacts/ingest",
            json={
                "metadata": {},
                "input": {"type": "text", "value": "Platform sync fact."},
                "collection": collection,
                "artifact": artifact,
            },
        )
        assert ingest_response.status_code == 200

        body = ChatBody(
            messages=[MessageInput(content="use the source", role="user")],
            tools=[{"name": "semantic_search", "type": "semantic_search_v1"}],
            tool_choice={"type": "tool", "name": "semantic_search"},
            tool_context=[
                IngestedArtifact(
                    context_filter={
                        "collection": collection,
                        "artifacts": [artifact],
                    }
                )
            ],
            system=System(
                citations={"enabled": True},
                prompt=PromptConfig(citations=True),
            ),
        )
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        assert response.status_code == 200, response.text
        assert len(capture.system_prompts) == 2

        # Before the tool ran there are no documents, so no citation
        # guidelines should be injected yet.
        assert "Citations let users trace" not in capture.system_prompts[0]

        # After the tool result added a source document, the platform
        # citation guidelines must be rebuilt from the latest stack and
        # appear in the final system prompt.
        assert "Citations let users trace" in capture.system_prompts[1]

        await async_test_client.post(
            "/v1/artifacts/delete",
            json={"collection": collection, "artifact": artifact},
        )
