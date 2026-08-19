"""Lifecycle invariants for platform prompts across iterations and resume.

Inputs persist. Derived layers are rebuilt every iteration from those inputs
plus the current messages:

- USER_INSTRUCTIONS     input: original request.system.prompt (restore)
- RUNTIME_INSTRUCTIONS  derived: SystemPrompt header (remove + regenerate)
- TOOL_INSTRUCTIONS     derived: PlatformGuidelines if flags still on
- SKILL_CATALOG/BODY    derived: cache + load/unload history (never freeze)
- TOOL_DEFINITIONS      derived: request tools + MCP + internal + visibility
- original_input        input snapshot; never rewritten from a rendered prompt

Catalog/body/tools change when messages change (load_skill, deferred tools).
API-entry and in-loop continuation must produce the same *derived* prompt
for the same inputs + messages.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from pydantic import Field

from private_gpt.chat.input_models import PromptConfig
from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import (
    SkillBodyLayer,
    SkillCatalogEntry,
    SkillCatalogLayer,
    ToolDefinitionsLayer,
    ToolInstructionsLayer,
)
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat.async_chat_engine import (
    AsyncChatEngine,
    LocalEventChannel,
)
from private_gpt.components.engines.chat.checkpoint_store import ChatCheckpoint
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.interceptors.ensure_tools_are_flatten_interceptor import (
    EnsureToolAreFlattenInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
)
from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner
from private_gpt.components.engines.chat.utils.request_builder import (
    build_initial_context_stack,
    build_request_from_context_stack,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import (
    BASH_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
)
from private_gpt.components.tools.tool_scheduler import LocalToolScheduler
from private_gpt.server.chat.interceptors.platform_guidelines_interceptor import (
    PlatformGuidelinesInterceptor,
)
from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
    SkillsInterceptor,
)
from private_gpt.server.chat.interceptors.system_prompt_interceptor import (
    SystemPromptRequestInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

USER_PROMPT = "USER_PLATFORM_PROMPT: follow the user's instructions."
HEADER = "You are Zylon, an AI assistant.\nCurrent date: 2026-08-19."
SKILL_CATALOG_NAME = "skill-creator"
SKILL_BODY_MARKER = "<response_formatting>Be clear.</response_formatting>"
BASH_INSTRUCTIONS = "BASH TOOL INSTRUCTIONS"
CODE_EXEC_PROMPT = "CODE EXECUTION PLATFORM PROMPT"
SKILLS_PROMPT = "SKILLS MANAGEMENT PLATFORM PROMPT"
MCP_TOOL_NAME = "mcp_lookup"


def _count(text: str, marker: str) -> int:
    return len(re.findall(re.escape(marker), text))


def _system_text(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            continue
        if message.blocks:
            for block in message.blocks:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        elif message.content:
            parts.append(str(message.content))
    return "\n".join(parts)


def _prompt_builder() -> MagicMock:
    builder = MagicMock()

    def _template(text: str) -> MagicMock:
        template = MagicMock()
        template.format.return_value = text
        return template

    builder.create_chat_header_prompt.return_value = _template(HEADER)
    builder.create_code_execution_prompt.return_value = _template(CODE_EXEC_PROMPT)
    builder.create_skills_prompt.return_value = _template(SKILLS_PROMPT)
    builder.create_thinking_guidelines.return_value = _template("")
    builder.create_citation_guidelines.return_value = _template("")
    return builder


async def _echo_tool(value: str) -> str:
    return f"ok:{value}"


def _rebuild_echo(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name, type=name, runtime="server", async_fn=_echo_tool
    )


def _server_tools() -> list[ToolSpec]:
    return [
        ToolSpec.from_defaults(
            name="echo",
            type="echo",
            runtime="server",
            async_fn=_echo_tool,
            execution_metadata=build_rebuild_metadata(_rebuild_echo, {"name": "echo"}),
        ),
        ToolSpec.from_defaults(
            name=BASH_TOOL_NAME,
            type=BASH_TOOL_NAME,
            runtime="server",
            async_fn=_echo_tool,
            instructions=BASH_INSTRUCTIONS,
            execution_metadata=build_rebuild_metadata(
                _rebuild_echo, {"name": BASH_TOOL_NAME}
            ),
        ),
        ToolSpec.from_defaults(
            name=SKILL_LOAD_TOOL_NAME,
            type=SKILL_LOAD_TOOL_NAME,
            runtime="server",
            async_fn=_echo_tool,
            execution_metadata=build_rebuild_metadata(
                _rebuild_echo, {"name": SKILL_LOAD_TOOL_NAME}
            ),
        ),
    ]


class _FakeChatScheduler:
    async def cancel(self, correlation_id: str) -> bool:
        del correlation_id
        return True


class _SeedPlatformStateInterceptor(ChatRequestLoopInterceptor):
    """Inject skill/MCP layers. ``once`` seeds only on VALIDATION (cannot rebuild)."""

    once: bool = False
    seeded: bool = False

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if self.once and context.phase != InterceptorPhase.VALIDATION:
            return
        if not self.once and context.phase != InterceptorPhase.BEFORE_ITERATION:
            return
        if self.once and self.seeded:
            return

        stack = context.state.input.context_stack
        stack = stack.remove_layers_of_source("skills")
        stack = stack.remove_layers_of_source("mcp")
        stack = stack.append_layer(
            SkillCatalogLayer(
                entries=[
                    SkillCatalogEntry(
                        id="skill-creator",
                        name=SKILL_CATALOG_NAME,
                        description="Create new skills",
                        loading="lazy",
                    )
                ],
                source="skills",
            )
        )
        stack = stack.append_layer(
            SkillBodyLayer(
                skill_id="response-guidelines",
                name="response-guidelines",
                version="1",
                instructions=SKILL_BODY_MARKER,
                source="skills",
                render_as_xml=False,
            )
        )
        stack = stack.append_layer(
            ToolDefinitionsLayer(
                tools=[
                    ToolSpec.from_defaults(
                        name=MCP_TOOL_NAME,
                        type=MCP_TOOL_NAME,
                        runtime="client",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                source="mcp",
            )
        )
        context.state.input.context_stack = stack
        context.set_state(context.state)
        self.seeded = True


class _LlmCapture:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.tool_names: list[list[str]] = []
        self.histories: list[list[ChatMessage]] = []


def _capturing_llm(
    deltas: list[list[str | ToolSelection]], capture: _LlmCapture
) -> Any:
    mock_llm = get_mock_function_calling_llm(deltas)
    original = mock_llm.astream_chat_with_tools

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tools = kwargs.get("tools", args[0] if args else [])
        history = kwargs.get("chat_history")
        if history is None and len(args) >= 3:
            history = args[2]
        history = history or []
        capture.histories.append(list(history))
        capture.system_prompts.append(_system_text(history))
        names: list[str] = []
        for tool in tools or []:
            metadata = getattr(tool, "metadata", None)
            name = getattr(metadata, "name", None) if metadata is not None else None
            if name:
                names.append(name)
        capture.tool_names.append(names)
        return await original(*args, **kwargs)

    mock_llm.astream_chat_with_tools = wrapper
    return mock_llm


def _make_engine(
    mock_llm: Any,
    request_interceptors: list[ChatRequestLoopInterceptor],
) -> AsyncChatEngine:
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = mock_llm
    return AsyncChatEngine(
        llm_component=llm_component,
        request_interceptors=request_interceptors,
        response_interceptors=[],
        max_iterations=6,
        tool_scheduler=LocalToolScheduler(),
        chat_scheduler=_FakeChatScheduler(),
    )


def _lifecycle_interceptors(
    *,
    seed_once: bool = False,
    include_skills_interceptor: bool = False,
) -> list[ChatRequestLoopInterceptor]:
    builder = _prompt_builder()
    settings = MagicMock()
    settings.skills.skill_injection_mode = "system_prompt"
    interceptors: list[ChatRequestLoopInterceptor] = [
        _SeedPlatformStateInterceptor(once=seed_once),
        PlatformGuidelinesInterceptor(prompt_builder=builder, settings=settings),
        SystemPromptRequestInterceptor(
            prompt_builder_service=builder,
            add_context_to_system_prompt=False,
        ),
    ]
    if include_skills_interceptor:
        interceptors.insert(
            1,
            SkillsInterceptor(
                skill_service=MagicMock(),
                skill_loader=MagicMock(),
                settings=settings,
            ),
        )
    return interceptors


def _base_request() -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="help me")],
        system=ResolvedSystemConfig(
            prompt=USER_PROMPT,
            platform_prompts=PromptConfig(
                tools=True,
                skills=True,
                code_execution=True,
            ),
        ),
        tool_config=ResolvedToolConfig(tools=_server_tools()),
    )


def _tool_call(tool_id: str, name: str, value: str) -> ToolSelection:
    return ToolSelection(tool_id=tool_id, tool_name=name, tool_kwargs={"value": value})


async def _run_loop(
    request: ResolvedChatRequest,
    deltas: list[list[str | ToolSelection]],
    interceptors: list[ChatRequestLoopInterceptor],
) -> _LlmCapture:
    capture = _LlmCapture()
    engine = _make_engine(_capturing_llm(deltas, capture), interceptors)
    channel = LocalEventChannel()
    await engine.execute(request, channel=channel)
    await channel.close()
    async for _ in channel.stream():
        pass
    return capture


def _assert_single_markers(system_prompt: str) -> None:
    assert _count(system_prompt, USER_PROMPT) == 1, system_prompt
    assert _count(system_prompt, HEADER) == 1, system_prompt
    assert _count(system_prompt, SKILL_CATALOG_NAME) == 1, system_prompt
    assert _count(system_prompt, SKILL_BODY_MARKER) == 1, system_prompt
    assert _count(system_prompt, BASH_INSTRUCTIONS) == 1, system_prompt
    assert _count(system_prompt, CODE_EXEC_PROMPT) == 1, system_prompt
    assert _count(system_prompt, SKILLS_PROMPT) == 1, system_prompt


@pytest.mark.asyncio
async def test_no_duplication_when_iteration_continues() -> None:
    capture = await _run_loop(
        _base_request(),
        [
            [_tool_call("t1", "echo", "a")],
            [_tool_call("t2", "echo", "b")],
            ["done"],
        ],
        _lifecycle_interceptors(),
    )

    assert len(capture.system_prompts) == 3
    for prompt in capture.system_prompts:
        _assert_single_markers(prompt)


@pytest.mark.asyncio
async def test_llm_system_prompt_equal_across_iterations() -> None:
    capture = await _run_loop(
        _base_request(),
        [
            [_tool_call("t1", "echo", "a")],
            ["done"],
        ],
        _lifecycle_interceptors(),
    )

    assert len(capture.system_prompts) == 2
    assert capture.system_prompts[0] == capture.system_prompts[1]
    _assert_single_markers(capture.system_prompts[0])


@pytest.mark.asyncio
async def test_fresh_api_request_matches_continued_iteration() -> None:
    """A new API call with the same original system + history must match loop iter N."""
    interceptors = _lifecycle_interceptors()
    continued = await _run_loop(
        _base_request(),
        [
            [_tool_call("t1", "echo", "a")],
            ["done"],
        ],
        interceptors,
    )
    assert len(continued.system_prompts) == 2

    fresh_request = _base_request()
    # Replay the non-system conversation from the continued run's second call.
    continued_history = [
        message
        for message in continued.histories[1]
        if message.role != MessageRole.SYSTEM
    ]
    fresh_request.messages = list(continued_history)
    fresh = await _run_loop(
        fresh_request,
        [["ok"]],
        _lifecycle_interceptors(),
    )

    assert len(fresh.system_prompts) == 1
    assert fresh.system_prompts[0] == continued.system_prompts[1]
    _assert_single_markers(fresh.system_prompts[0])


class _SeedSkillCacheInterceptor(ChatRequestLoopInterceptor):
    """Stand-in for SkillsValidationInterceptor: populate cache once."""

    entries: list[Any] = Field(default_factory=list)

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.VALIDATION:
            return
        from private_gpt.components.engines.chat.models.chat_state import (
            SkillsRuntimeCache,
        )

        context.state.runtime.cache.skill = SkillsRuntimeCache(
            entries=list(self.entries)
        )
        context.set_state(context.state)


@pytest.mark.asyncio
async def test_skill_catalog_becomes_body_after_load_skill_iteration() -> None:
    """Catalog/body are recomputed from messages: load moves a skill out of catalog."""
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock

    from private_gpt.components.skills.models.skill_entities import (
        SkillEntity,
        SkillFilter,
        SkillFrontmatter,
        SkillVersionEntity,
        SkillVersionWithSkillEntity,
    )
    from private_gpt.server.utils.artifact_input import SkillArtifact

    now = datetime.now(UTC)
    entry = SkillVersionWithSkillEntity(
        skill=SkillEntity(
            id="skill-creator",
            collection="col",
            display_title="skill-creator",
            source="zylon",
            loading="lazy",
            readonly=True,
            created_at=now,
            updated_at=now,
        ),
        version=SkillVersionEntity(
            id="ver-creator",
            skill_id="skill-creator",
            version="1",
            frontmatter=SkillFrontmatter(
                name="skill-creator", description="Create skills"
            ),
            storage_prefix="skills/creator",
            created_at=now,
        ),
    )

    async def _load_skill_tool(name: str) -> str:
        return f'{{"name": "{name}", "loaded": true}}'

    def _rebuild_load(name: str) -> ToolSpec:
        return ToolSpec.from_defaults(
            name=name, type=name, runtime="server", async_fn=_load_skill_tool
        )

    request = _base_request()
    request.tool_context = [SkillArtifact(skill_filter=SkillFilter(collection="col"))]
    request.tool_config.tools = [
        tool for tool in request.tool_config.tools if tool.name != SKILL_LOAD_TOOL_NAME
    ] + [
        ToolSpec.from_defaults(
            name=SKILL_LOAD_TOOL_NAME,
            type=SKILL_LOAD_TOOL_NAME,
            runtime="server",
            async_fn=_load_skill_tool,
            execution_metadata=build_rebuild_metadata(
                _rebuild_load, {"name": SKILL_LOAD_TOOL_NAME}
            ),
        )
    ]

    builder = _prompt_builder()
    settings = MagicMock()
    settings.skills.skill_injection_mode = "system_prompt"
    skill_service = MagicMock()
    skill_service.get_skill_body = AsyncMock(return_value="CREATOR BODY")
    skill_loader = MagicMock()
    skill_loader.mounts_for_versions.return_value = []

    capture = await _run_loop(
        request,
        [
            [
                ToolSelection(
                    tool_id="t1",
                    tool_name=SKILL_LOAD_TOOL_NAME,
                    tool_kwargs={"name": "skill-creator"},
                )
            ],
            ["done"],
        ],
        [
            _SeedSkillCacheInterceptor(entries=[entry]),
            SkillsInterceptor(
                skill_service=skill_service,
                skill_loader=skill_loader,
                settings=settings,
            ),
            PlatformGuidelinesInterceptor(prompt_builder=builder, settings=settings),
            SystemPromptRequestInterceptor(
                prompt_builder_service=builder,
                add_context_to_system_prompt=False,
            ),
        ],
    )

    assert len(capture.system_prompts) == 2
    assert "<available_skills>" in capture.system_prompts[0]
    assert "CREATOR BODY" not in capture.system_prompts[0]
    assert "<available_skills>" not in capture.system_prompts[1]
    assert "CREATOR BODY" in capture.system_prompts[1]
    # Header/user prompt stay stable and non-duplicated.
    assert _count(capture.system_prompts[0], USER_PROMPT) == 1
    assert _count(capture.system_prompts[1], USER_PROMPT) == 1
    assert _count(capture.system_prompts[0], HEADER) == 1
    assert _count(capture.system_prompts[1], HEADER) == 1


@pytest.mark.asyncio
async def test_enabled_tools_present_on_every_iteration() -> None:
    capture = await _run_loop(
        _base_request(),
        [
            [_tool_call("t1", "echo", "a")],
            ["done"],
        ],
        _lifecycle_interceptors(),
    )

    assert len(capture.tool_names) == 2
    for names in capture.tool_names:
        assert "echo" in names
        assert BASH_TOOL_NAME in names
        assert SKILL_LOAD_TOOL_NAME in names
        assert MCP_TOOL_NAME in names
    assert capture.tool_names[0] == capture.tool_names[1]


def test_materialized_request_keeps_platform_prompt_flags() -> None:
    request = _base_request()
    stack = build_initial_context_stack(request)
    stack = stack.append_layer(
        ToolInstructionsLayer(
            tool_name="bash",
            instructions=CODE_EXEC_PROMPT,
            source="platform:code_execution",
        )
    )
    materialized = build_request_from_context_stack(request, stack)
    assert materialized.system.platform_prompts.tools is True
    assert materialized.system.platform_prompts.skills is True
    assert materialized.system.platform_prompts.code_execution is True


def test_original_input_not_poisoned_by_rendered_prompt() -> None:
    request = _base_request()
    llm_component = MagicMock(spec=LLMComponent)
    llm_component.get_llm.return_value = get_mock_function_calling_llm(["ok"])
    engine = AsyncChatEngine(
        llm_component=llm_component,
        chat_scheduler=_FakeChatScheduler(),
    )
    first = engine.initialize_run(request)
    # Simulate the engine writing the full stack back into system.prompt.
    full = build_request_from_context_stack(
        first.state.input.request,
        first.state.input.context_stack.append_layer(
            SkillBodyLayer(
                skill_id="rg",
                name="response-guidelines",
                version="1",
                instructions=SKILL_BODY_MARKER,
                source="skills",
                render_as_xml=False,
            )
        ),
    )
    second = engine.initialize_run(
        full,
        context_stack=first.state.input.context_stack,
        original_input=first.state.original_input,
    )
    assert second.state.original_input is first.state.original_input
    original_layers = second.state.original_input.context_stack.layers_of_type(
        LayerType.USER_INSTRUCTIONS
    )
    assert original_layers
    assert SKILL_BODY_MARKER not in original_layers[0].render()
    assert USER_PROMPT in original_layers[0].render()


def test_checkpoint_roundtrip_keeps_original_user_prompt() -> None:
    request = _base_request()
    original = ChatInputState(
        request=request,
        context_stack=build_initial_context_stack(request),
    )
    rendered_stack = original.context_stack.append_layer(
        SkillCatalogLayer(
            entries=[
                SkillCatalogEntry(
                    id="1",
                    name=SKILL_CATALOG_NAME,
                    description="Create skills",
                    loading="lazy",
                )
            ],
            source="skills",
        )
    )
    rendered_request = build_request_from_context_stack(request, rendered_stack)
    checkpoint = ChatCheckpoint(
        correlation_id="exec-1",
        request_data=rendered_request.model_dump(mode="json"),
        context_stack_data=rendered_stack.checkpoint_dump(),
        original_input_data=ResumableChatRunner._dump_original_input(original),
        stream_type="chat_completion",
        metadata={},
        iteration=1,
    )

    restored_original = ResumableChatRunner._original_input(checkpoint)
    restored_stack = ResumableChatRunner._context_stack(
        checkpoint, checkpoint.request_data
    )
    assert restored_original is not None
    user_layers = restored_original.context_stack.layers_of_type(
        LayerType.USER_INSTRUCTIONS
    )
    assert user_layers
    assert user_layers[0].render() == USER_PROMPT
    assert SKILL_CATALOG_NAME not in user_layers[0].render()
    assert restored_stack.layers_of_type(LayerType.SKILL_CATALOG)
    resumed_request = ResumableChatRunner._request(checkpoint.request_data)
    assert resumed_request.system.platform_prompts.skills is True
    assert resumed_request.system.platform_prompts.code_execution is True


def test_rebuild_from_mutated_request_without_stack_collapses_layers() -> None:
    """Documents the poison path: never rebuild the stack from a materialized prompt."""
    request = _base_request()
    stack = build_initial_context_stack(request).append_layer(
        SkillCatalogLayer(
            entries=[
                SkillCatalogEntry(
                    id="1",
                    name=SKILL_CATALOG_NAME,
                    description="Create skills",
                    loading="lazy",
                )
            ],
            source="skills",
        )
    )
    mutated = build_request_from_context_stack(request, stack)
    collapsed = build_initial_context_stack(mutated)
    types = [layer.type for layer in collapsed.layers]
    assert LayerType.SKILL_CATALOG not in types
    assert LayerType.USER_INSTRUCTIONS in types
    assert (
        SKILL_CATALOG_NAME
        in collapsed.layers_of_type(LayerType.USER_INSTRUCTIONS)[0].render()
    )


@pytest.mark.asyncio
async def test_restore_keeps_mcp_tools_not_in_original_snapshot() -> None:
    from private_gpt.components.context.models.context_layer import (
        UserInstructionsLayer,
    )
    from private_gpt.components.engines.chat.interceptors.restore_stateless_input_interceptor import (
        RestoreStatelessInputInterceptorRequest,
    )
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatOutputState,
        ChatRuntimeState,
        ChatState,
    )

    original_stack = ContextStack(
        layers=[
            UserInstructionsLayer(text=USER_PROMPT, source="request"),
            ToolDefinitionsLayer(tools=_server_tools(), source="request"),
        ]
    )
    current_stack = original_stack.append_layer(
        ToolDefinitionsLayer(
            tools=[
                ToolSpec.from_defaults(
                    name=MCP_TOOL_NAME,
                    type=MCP_TOOL_NAME,
                    runtime="client",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            source="mcp",
        )
    )
    request = _base_request()
    state = ChatState(
        input=ChatInputState(request=request, context_stack=current_stack),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        timeline=[],
        original_input=ChatInputState(request=request, context_stack=original_stack),
    )
    context = ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.BEFORE_ITERATION,
        emit_fn=lambda _: None,
    )
    await RestoreStatelessInputInterceptorRequest().intercept(context)
    names = [
        tool.name for tool in context.state.input.context_stack.all_tools() if tool.name
    ]
    assert "echo" in names
    assert MCP_TOOL_NAME in names


def _doc(doc_id: str, text: str, shorter_id: str) -> Any:
    from private_gpt.components.engines.citations.types import Document

    return Document(
        id_=doc_id,
        type="document",
        text=text,
        shorter_id=shorter_id,
        document_id=doc_id,
        _metadata={"shorter_id": shorter_id},
    )


@pytest.mark.asyncio
async def test_citation_extractor_uses_prompt_document_snapshot() -> None:
    """Extractor must keep the prompt's document set even if the stack changes."""

    from private_gpt.components.chat.models.chat_config_models import CitationConfig
    from private_gpt.components.context.models.context_layer import DocumentLayer
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatOutputState,
        ChatRuntimeState,
        ChatState,
    )
    from private_gpt.server.chat.interceptors.extract_citation_interceptor import (
        ExtractCitationInterceptor,
    )

    prompt_doc = _doc("doc_paris_001", "Paris is the capital of France.", "ab12")
    later_doc = _doc("doc_lyon_002", "Lyon is in France.", "cd34")
    request = _base_request()
    request.citation = CitationConfig(enabled=True)
    stack = ContextStack(
        layers=[DocumentLayer(document=prompt_doc, source="citations")]
    )
    state = ChatState(
        input=ChatInputState(request=request, context_stack=stack),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        timeline=[],
    )
    interceptor = ExtractCitationInterceptor()
    context = ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.STREAMING,
        emit_fn=lambda _: None,
    )
    await interceptor.on_iteration_start(context)
    snapped = interceptor._documents_for_prompt(context)
    assert [doc.id_ for doc in snapped] == ["doc_paris_001"]

    # Stack changes after the prompt was built (should not affect this iteration).
    context.state.input.context_stack = ContextStack(
        layers=[DocumentLayer(document=later_doc, source="citations")]
    )
    assert [doc.id_ for doc in interceptor._documents_for_prompt(context)] == [
        "doc_paris_001"
    ]


@pytest.mark.asyncio
async def test_citation_interceptor_merges_request_docs_with_history_sources() -> None:
    """Request documents stay; new tool sources from history are appended."""
    from llama_index.core.schema import NodeWithScore, TextNode

    from private_gpt.components.chat.models.chat_config_models import CitationConfig
    from private_gpt.components.context.models.context_layer import DocumentLayer
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatOutputState,
        ChatRuntimeState,
        ChatState,
    )
    from private_gpt.events.models import SourceBlock
    from private_gpt.server.chat.interceptors.citation_interceptor import (
        CitationRequestInterceptor,
    )

    request_doc = _doc("doc_req", "Request document.", "aa11")
    history_node = TextNode(
        text="Paris is the capital of France.",
        id_="doc_paris_001",
        metadata={"shorter_id": "ab12", "source_id": "src_paris"},
    )
    request = _base_request()
    request.citation = CitationConfig(enabled=True)
    request.messages = [
        ChatMessage(role=MessageRole.USER, content="where is paris"),
        ChatMessage(
            role=MessageRole.TOOL,
            content="hits",
            additional_kwargs={
                "source": [
                    SourceBlock.from_nodes(
                        [NodeWithScore(node=history_node, score=0.9)]
                    )
                ]
            },
        ),
    ]
    state = ChatState(
        input=ChatInputState(
            request=request,
            context_stack=ContextStack(
                layers=[DocumentLayer(document=request_doc, source="request")]
            ),
        ),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        timeline=[],
    )
    context = ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.BEFORE_ITERATION,
        emit_fn=lambda _: None,
    )
    await CitationRequestInterceptor().intercept(context)
    ids = [doc.id_ for doc in context.state.input.context_stack.all_documents()]
    assert "doc_req" in ids
    assert "doc_paris_001" in ids


@pytest.mark.asyncio
async def test_tool_flattening_does_not_mutate_original_input_messages() -> None:
    tool_id = "tool-1"
    assistant = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        additional_kwargs={
            "tool_calls": [
                ToolSelection(
                    tool_id=tool_id,
                    tool_name="load_skill",
                    tool_kwargs={"name": "skill-creator"},
                )
            ]
        },
    )
    tool = ChatMessage(
        role=MessageRole.TOOL,
        content='{"name":"skill-creator","loaded":true}',
        additional_kwargs={
            "tool_call_id": tool_id,
            "tool_call_name": "load_skill",
            "tool_call_args": {"name": "skill-creator"},
        },
    )
    user = ChatMessage(role=MessageRole.USER, content="continue")

    request = _base_request()
    request.messages = [assistant, tool, user]

    state = ChatState(
        input=ChatInputState(request=request),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        original_input=ChatInputState(
            request=request.model_copy(deep=True),
        ),
    )

    original_messages = list(state.original_input.request.messages)
    context = ChatInterceptorContext(
        state=state,
        llm=FunctionCallingLLM.model_construct(),
        phase=InterceptorPhase.AFTER_ITERATION,
        emit_fn=lambda _event: None,
    )

    await EnsureToolAreFlattenInterceptor().intercept(context)

    assert state.input.request.messages != original_messages
    assert state.original_input.request.messages == original_messages
