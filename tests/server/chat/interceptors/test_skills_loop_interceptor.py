from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.tools.tool_names import (
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.events.models import NO_TOOL_CONTENT
from private_gpt.server.chat.interceptors.server_tool_result_text_interceptor import (
    ServerToolResultTextInterceptor,
)
from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
    _resolve_active_skill_names,
    _resolve_skill_states,
    find_skill_filter,
    partition_skill_entries,
)


def _tool_message(
    *,
    call_name: str,
    content: str | None,
    args: dict[str, str] | None = None,
) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.TOOL,
        content=content,
        additional_kwargs={
            "tool_call_name": call_name,
            "tool_call_args": args or {},
        },
    )


def test_resolve_skill_states_from_load_result_json() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "loaded": true}',
            args={"name": "skill-creator"},
        )
    ]

    assert _resolve_active_skill_names(messages) == {"skill-creator"}


def test_resolve_skill_states_falls_back_to_tool_call_args() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content=NO_TOOL_CONTENT,
            args={"name": "skill-creator"},
        )
    ]

    assert _resolve_active_skill_names(messages) == {"skill-creator"}


def test_resolve_skill_states_ignores_failed_load_json() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "error": "missing"}',
            args={"name": "skill-creator"},
        )
    ]

    assert _resolve_active_skill_names(messages) == set()


def test_resolve_skill_states_unload_from_json() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "loaded": true}',
            args={"name": "skill-creator"},
        ),
        _tool_message(
            call_name=SKILL_UNLOAD_TOOL_NAME,
            content='{"name": "skill-creator", "unloaded": true}',
            args={"name": "skill-creator"},
        ),
    ]

    active, removed = _resolve_skill_states(messages)
    assert active == set()
    assert removed == {"skill-creator"}


def test_resolve_skill_states_unload_falls_back_to_tool_call_args() -> None:
    messages = [
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "loaded": true}',
            args={"name": "skill-creator"},
        ),
        _tool_message(
            call_name=SKILL_UNLOAD_TOOL_NAME,
            content=NO_TOOL_CONTENT,
            args={"name": "skill-creator"},
        ),
    ]

    active, removed = _resolve_skill_states(messages)
    assert active == set()
    assert removed == {"skill-creator"}


def test_resolve_skill_states_from_assistant_tool_calls_without_tool_result() -> None:
    messages = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I'll load skill-creator.",
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id="srvtoolu_load_skill_1",
                        tool_name=SKILL_LOAD_TOOL_NAME,
                        tool_kwargs={"name": "skill-creator"},
                    )
                ]
            },
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Great! Let me help you create a skill.",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Can you draft a version based on that?",
        ),
    ]

    assert _resolve_active_skill_names(messages) == {"skill-creator"}


def test_resolve_skill_states_failed_load_result_overrides_assistant_tool_call() -> (
    None
):
    messages = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id="tu_load",
                        tool_name=SKILL_LOAD_TOOL_NAME,
                        tool_kwargs={"name": "skill-creator"},
                    )
                ]
            },
        ),
        _tool_message(
            call_name=SKILL_LOAD_TOOL_NAME,
            content='{"name": "skill-creator", "error": "missing"}',
            args={"name": "skill-creator"},
        ),
    ]

    assert _resolve_active_skill_names(messages) == set()


def test_resolve_skill_states_from_user_tool_response_json() -> None:
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content='<tool_response>\n{"name": "skill-creator", "loaded": true}\n</tool_response>',
        )
    ]

    assert _resolve_active_skill_names(messages) == {"skill-creator"}


_CLIENT_FOLLOWUP_HISTORY = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Create a skill for building internal metrics dashboards.",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll help you create a skill for building internal metrics dashboards. Let me first check what skills are available and then create a new one tailored to your needs.\n\n",
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_list_skills_1",
                "name": "list_skills",
                "input": {"page": 0, "page_size": 20},
            },
            {
                "type": "server_tool_result",
                "tool_use_id": "srvtoolu_list_skills_1",
                "content": [
                    {
                        "type": "text",
                        "text": '{"skills": [{"name": "skill-creator", "description": "Create new skills"}], "page": 0, "page_size": 20, "total": 1, "has_more": false}',
                    }
                ],
                "is_error": False,
            },
            {
                "type": "text",
                "text": "I'll help you create a skill for building internal metrics dashboards. Let me start by loading the skill-creator tool which will allow us to build this new capability.\n\n",
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_load_skill_1",
                "name": "load_skill",
                "input": {"name": "skill-creator"},
            },
            {
                "type": "server_tool_result",
                "tool_use_id": "srvtoolu_load_skill_1",
                "content": [
                    {
                        "type": "text",
                        "text": '{"name": "skill-creator", "loaded": true}',
                    }
                ],
                "is_error": False,
            },
            {
                "type": "text",
                "text": "Great! Let me help you create a skill for building internal metrics dashboards.",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Let's go with a mix of technical and operational metrics. Can you draft a version based on that?",
            }
        ],
    },
]


def test_client_history_with_server_tool_load_keeps_skill_creator_loaded() -> None:
    messages = MessageInput.convert_from_llama_index_messages(
        [MessageInput.model_validate(message) for message in _CLIENT_FOLLOWUP_HISTORY]
    )

    load_msgs = [
        msg
        for msg in messages
        if msg.role == MessageRole.TOOL
        and msg.additional_kwargs.get("tool_call_name") == SKILL_LOAD_TOOL_NAME
    ]
    assert load_msgs, (
        "load_skill server_tool_result must become a TOOL message with tool_call_name"
    )
    assert _resolve_active_skill_names(messages) == {"skill-creator"}


@pytest.mark.anyio
async def test_client_history_still_loaded_after_server_tool_result_rewrite() -> None:
    messages = MessageInput.convert_from_llama_index_messages(
        [MessageInput.model_validate(message) for message in _CLIENT_FOLLOWUP_HISTORY]
    )
    state = SimpleNamespace(
        input=SimpleNamespace(request=SimpleNamespace(messages=messages))
    )
    context = SimpleNamespace(phase=InterceptorPhase.BEFORE_ITERATION, state=state)
    context.set_state = lambda new_state: None

    await ServerToolResultTextInterceptor().intercept(context)

    assert _resolve_active_skill_names(context.state.input.request.messages) == {
        "skill-creator"
    }


def _skill_cache_entry(*, name: str, loading: str = "lazy"):
    from datetime import UTC, datetime

    from private_gpt.components.skills.models.skill_entities import (
        SkillEntity,
        SkillFrontmatter,
        SkillVersionEntity,
        SkillVersionWithSkillEntity,
    )

    now = datetime.now(UTC)
    skill = SkillEntity(
        id=f"skill-{name}",
        collection="col",
        display_title=name,
        source="zylon",
        loading=loading,  # type: ignore[arg-type]
        readonly=True,
        created_at=now,
        updated_at=now,
    )
    version = SkillVersionEntity(
        id=f"ver-{name}",
        skill_id=skill.id,
        version="1",
        frontmatter=SkillFrontmatter(name=name, description=f"{name} description"),
        storage_prefix=f"skills/{name}",
        created_at=now,
    )
    return SkillVersionWithSkillEntity(skill=skill, version=version)


def _skills_interceptor(*, body: str = "SKILL BODY"):
    from unittest.mock import AsyncMock, MagicMock

    from private_gpt.server.chat.interceptors.skills_loop_interceptor import (
        SkillsInterceptor,
    )

    skill_service = MagicMock()
    skill_service.get_skill_body = AsyncMock(return_value=body)
    skill_loader = MagicMock()
    skill_loader.mounts_for_versions.return_value = []
    return SkillsInterceptor(
        skill_service=skill_service,
        skill_loader=skill_loader,
        settings=MagicMock(skills=MagicMock(skill_injection_mode="system_prompt")),
    )


def _skill_request(messages: list[ChatMessage]):
    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
        ResolvedSystemConfig,
    )
    from private_gpt.components.skills.models.skill_entities import SkillFilter
    from private_gpt.server.utils.artifact_input import SkillArtifact

    return ResolvedChatRequest(
        messages=messages,
        system=ResolvedSystemConfig(prompt="You are Zylon"),
        tool_context=[SkillArtifact(skill_filter=SkillFilter(collection="col"))],
    )


def _skill_context(
    request,
    *,
    names: list[str],
    messages: list[ChatMessage] | None = None,
    loadings: dict[str, str] | None = None,
    resources: dict[str, list[str]] | None = None,
):
    from private_gpt.components.context.models.context_stack import ContextStack
    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatInputState,
        ChatOutputState,
        ChatRuntimeCache,
        ChatRuntimeState,
        ChatState,
        SkillsRuntimeCache,
    )
    from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

    if messages is not None:
        request = request.model_copy(deep=True)
        request.messages = messages
    state = ChatState(
        input=ChatInputState(request=request, context_stack=ContextStack()),
        runtime=ChatRuntimeState(
            cache=ChatRuntimeCache(
                skill=SkillsRuntimeCache(
                    entries=[
                        _skill_cache_entry(
                            name=name,
                            loading=(loadings or {}).get(name, "lazy"),
                        )
                        for name in names
                    ],
                    resources=resources or {},
                )
            )
        ),
        output=ChatOutputState(),
        timeline=[],
    )
    return ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.BEFORE_ITERATION,
        emit_fn=lambda _: None,
    )


@pytest.mark.anyio
async def test_skills_interceptor_without_filter_strips_stale_layers() -> None:
    """No skill context means catalog/body are not part of this request."""
    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
        ResolvedSystemConfig,
    )
    from private_gpt.components.context.models.context_layer import (
        SkillBodyLayer,
        SkillCatalogEntry,
        SkillCatalogLayer,
    )
    from private_gpt.components.context.models.context_stack import ContextStack
    from private_gpt.components.context.models.layer_type import LayerType
    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatInputState,
        ChatOutputState,
        ChatRuntimeState,
        ChatState,
    )
    from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

    stack = ContextStack(
        layers=[
            SkillCatalogLayer(
                entries=[
                    SkillCatalogEntry(
                        id="skill-creator",
                        name="skill-creator",
                        description="Create new skills",
                        loading="lazy",
                    )
                ],
                source="skills",
            ),
            SkillBodyLayer(
                skill_id="response-guidelines",
                name="response-guidelines",
                version="1",
                instructions="stale body",
                source="skill:response-guidelines",
                render_as_xml=False,
            ),
        ]
    )
    context = ChatInterceptorContext(
        state=ChatState(
            input=ChatInputState(
                request=ResolvedChatRequest(
                    messages=[ChatMessage(role=MessageRole.USER, content="hello")],
                    system=ResolvedSystemConfig(prompt="You are Zylon"),
                ),
                context_stack=stack,
            ),
            runtime=ChatRuntimeState(),
            output=ChatOutputState(),
            timeline=[],
        ),
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.BEFORE_ITERATION,
        emit_fn=lambda _: None,
    )

    await _skills_interceptor().intercept(context)

    stack = context.state.input.context_stack
    assert not stack.layers_of_type(LayerType.SKILL_CATALOG)
    assert not stack.layers_of_type(LayerType.SKILL_BODY)


@pytest.mark.anyio
async def test_skill_catalog_and_body_follow_load_and_unload_messages() -> None:
    """Catalog/body are derived from cache + messages, not frozen across turns."""
    from private_gpt.components.context.models.layer_type import LayerType

    request = _skill_request(
        [ChatMessage(role=MessageRole.USER, content="create a skill")]
    )
    interceptor = _skills_interceptor(body="CREATOR BODY")

    before_load = _skill_context(request, names=["skill-creator"])
    await interceptor.intercept(before_load)
    stack = before_load.state.input.context_stack
    catalog = stack.layers_of_type(LayerType.SKILL_CATALOG)
    body = stack.layers_of_type(LayerType.SKILL_BODY)
    assert catalog
    assert "skill-creator" in catalog[0].render()
    assert not body

    after_load = _skill_context(
        request,
        names=["skill-creator"],
        messages=[
            ChatMessage(role=MessageRole.USER, content="create a skill"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="tu_load",
                            tool_name=SKILL_LOAD_TOOL_NAME,
                            tool_kwargs={"name": "skill-creator"},
                        )
                    ]
                },
            ),
            _tool_message(
                call_name=SKILL_LOAD_TOOL_NAME,
                content='{"name": "skill-creator", "loaded": true}',
                args={"name": "skill-creator"},
            ),
        ],
    )
    await interceptor.intercept(after_load)
    stack = after_load.state.input.context_stack
    catalog = stack.layers_of_type(LayerType.SKILL_CATALOG)
    body = stack.layers_of_type(LayerType.SKILL_BODY)
    assert not catalog
    assert body
    assert "CREATOR BODY" in body[0].render()

    after_unload = _skill_context(
        request,
        names=["skill-creator"],
        messages=[
            *after_load.state.input.request.messages,
            ChatMessage(
                role=MessageRole.ASSISTANT,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="tu_unload",
                            tool_name=SKILL_UNLOAD_TOOL_NAME,
                            tool_kwargs={"name": "skill-creator"},
                        )
                    ]
                },
            ),
            _tool_message(
                call_name=SKILL_UNLOAD_TOOL_NAME,
                content='{"name": "skill-creator", "unloaded": true}',
                args={"name": "skill-creator"},
            ),
        ],
    )
    await interceptor.intercept(after_unload)
    stack = after_unload.state.input.context_stack
    catalog = stack.layers_of_type(LayerType.SKILL_CATALOG)
    body = stack.layers_of_type(LayerType.SKILL_BODY)
    assert catalog
    assert "skill-creator" in catalog[0].render()
    assert not body


def test_partition_skill_entries_eager_is_loaded_lazy_is_catalog() -> None:
    eager = _skill_cache_entry(name="guidelines", loading="eager")
    lazy = _skill_cache_entry(name="skill-creator", loading="lazy")
    partition = partition_skill_entries([eager, lazy], active_names=set())

    assert [v.frontmatter.name for v in partition.eager_versions] == ["guidelines"]
    assert partition.active_lazy_versions == []
    assert partition.loaded_names == {"guidelines"}
    assert [entry.name for entry in partition.catalog_entries] == ["skill-creator"]


def test_find_skill_filter_reads_tool_context() -> None:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec
    from private_gpt.components.skills.models.skill_entities import SkillFilter
    from private_gpt.server.utils.artifact_input import SkillArtifact

    artifact = SkillArtifact(skill_filter=SkillFilter(collection="from-tool"))
    tool = ToolSpec(name="list_skills", type="list_skills_v1", context=[artifact])

    assert find_skill_filter(tool_context=[], tools=[tool]) is artifact.skill_filter
    assert find_skill_filter(tool_context=None, tools=None) is None


@pytest.mark.anyio
async def test_eager_skill_is_injected_and_mounted_without_load() -> None:
    from private_gpt.components.context.models.layer_type import LayerType
    from private_gpt.components.skills.paths import skill_mount_path

    request = _skill_request(
        [ChatMessage(role=MessageRole.USER, content="create a skill")]
    )
    interceptor = _skills_interceptor(body="EAGER BODY")

    context = _skill_context(
        request,
        names=["guidelines", "skill-creator"],
        loadings={"guidelines": "eager", "skill-creator": "lazy"},
        resources={"skill-guidelines": ["scripts/init.py"]},
    )
    await interceptor.intercept(context)

    stack = context.state.input.context_stack
    catalog = stack.layers_of_type(LayerType.SKILL_CATALOG)
    body = stack.layers_of_type(LayerType.SKILL_BODY)
    assert catalog
    assert "skill-creator" in catalog[0].render()
    assert "guidelines" not in catalog[0].render()
    assert len(body) == 1
    rendered = body[0].render()
    assert "EAGER BODY" in rendered
    assert skill_mount_path("guidelines") in rendered
    assert "<skill_content" not in rendered

    mounted = interceptor._skill_loader.mounts_for_versions.call_args[0][0]
    assert [version.frontmatter.name for version in mounted] == ["guidelines"]
