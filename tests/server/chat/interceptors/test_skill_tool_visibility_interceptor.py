import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.tool_names import (
    SKILL_LIST_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
)
from private_gpt.server.chat.interceptors.skill_tool_visibility_interceptor import (
    SkillToolVisibilityInterceptor,
)


@pytest.fixture
def interceptor() -> SkillToolVisibilityInterceptor:
    return SkillToolVisibilityInterceptor()


@pytest.mark.parametrize(
    ("name", "tool_type", "expected_tokens"),
    [
        # versioned type produces both the versioned and unversioned token
        ("web_search", "web_search_v1", {"web_search", "web_search_v1"}),
        ("web_search", "web_search_v12", {"web_search", "web_search_v12"}),
        ("load_skill", "load_skill_v1", {"load_skill", "load_skill_v1"}),
        ("unload_skill", "unload_skill_v1", {"unload_skill", "unload_skill_v1"}),
        # unversioned type: stripping produces the same string, so only one entry
        ("web_search", "web_search", {"web_search"}),
        # name only (no type)
        ("semantic_search", None, {"semantic_search"}),
        # whitespace and casing are normalised
        (" Web_Search ", "Web_Search_V1", {"web_search", "web_search_v1"}),
    ],
)
def test_tool_tokens(
    interceptor: SkillToolVisibilityInterceptor,
    name: str | None,
    tool_type: str | None,
    expected_tokens: set[str],
) -> None:
    tool = ToolSpec(name=name, type=tool_type)
    assert interceptor._tool_tokens(tool) == expected_tokens


@pytest.mark.parametrize(
    ("has_loaded_skill", "has_catalog", "has_loaded_non_eager", "expected_visible"),
    [
        (False, False, False, set()),
        (False, True, False, {SKILL_LOAD_TOOL_NAME, SKILL_LIST_TOOL_NAME}),
        (True, True, False, {SKILL_LOAD_TOOL_NAME, SKILL_LIST_TOOL_NAME}),
        (True, False, True, {SKILL_UNLOAD_TOOL_NAME}),
        (
            True,
            True,
            True,
            {
                SKILL_LOAD_TOOL_NAME,
                SKILL_UNLOAD_TOOL_NAME,
                SKILL_LIST_TOOL_NAME,
            },
        ),
    ],
)
def test_skill_management_visibility_depends_on_catalog_and_loaded_non_eager(
    interceptor: SkillToolVisibilityInterceptor,
    has_loaded_skill: bool,
    has_catalog: bool,
    has_loaded_non_eager: bool,
    expected_visible: set[str],
) -> None:
    tools = [
        ToolSpec(name=SKILL_LOAD_TOOL_NAME, type="load_skill_v1"),
        ToolSpec(name=SKILL_UNLOAD_TOOL_NAME, type="unload_skill_v1"),
        ToolSpec(name=SKILL_LIST_TOOL_NAME, type="list_skills_v1"),
    ]

    visible = {
        tool.name
        for tool in tools
        if tool.name
        and interceptor._is_visible(
            tool=tool,
            has_loaded_skill=has_loaded_skill,
            has_catalog=has_catalog,
            has_loaded_non_eager=has_loaded_non_eager,
            allowed_tools={"some_other_tool"},
        )
    }

    assert visible == expected_visible


@pytest.mark.anyio
async def test_deferred_tool_appears_only_after_skill_is_loaded() -> None:
    """defer_loading tools are hidden until a skill is active in the messages."""
    from datetime import UTC, datetime

    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.llms.llm import ToolSelection

    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
        ResolvedSystemConfig,
        ResolvedToolConfig,
    )
    from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
    from private_gpt.components.context.models.context_stack import ContextStack
    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )
    from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatInputState,
        ChatOutputState,
        ChatRuntimeCache,
        ChatRuntimeState,
        ChatState,
        SkillsRuntimeCache,
    )
    from private_gpt.components.skills.models.skill_entities import (
        SkillEntity,
        SkillFilter,
        SkillFrontmatter,
        SkillVersionEntity,
        SkillVersionWithSkillEntity,
    )
    from private_gpt.server.utils.artifact_input import SkillArtifact
    from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

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
    deferred = ToolSpec.from_defaults(
        name="present_files",
        type="present_files",
        defer_loading=True,
        input_schema={"type": "object", "properties": {}},
    )
    always = ToolSpec.from_defaults(
        name="echo",
        type="echo",
        input_schema={"type": "object", "properties": {}},
    )

    def _ctx(messages: list[ChatMessage]) -> ChatInterceptorContext:
        request = ResolvedChatRequest(
            messages=messages,
            system=ResolvedSystemConfig(prompt="sys"),
            tool_config=ResolvedToolConfig(tools=[always, deferred]),
            tool_context=[SkillArtifact(skill_filter=SkillFilter(collection="col"))],
        )
        stack = ContextStack(
            layers=[ToolDefinitionsLayer(tools=[always, deferred], source="request")]
        )
        return ChatInterceptorContext(
            state=ChatState(
                input=ChatInputState(request=request, context_stack=stack),
                runtime=ChatRuntimeState(
                    cache=ChatRuntimeCache(skill=SkillsRuntimeCache(entries=[entry]))
                ),
                output=ChatOutputState(),
                timeline=[],
            ),
            llm=get_mock_function_calling_llm(["ok"]),
            phase=InterceptorPhase.BEFORE_ITERATION,
            emit_fn=lambda _: None,
        )

    interceptor = SkillToolVisibilityInterceptor()
    before = _ctx([ChatMessage(role=MessageRole.USER, content="hi")])
    await interceptor.intercept(before)
    assert {t.name for t in before.state.input.context_stack.all_tools()} == {"echo"}

    after = _ctx(
        [
            ChatMessage(role=MessageRole.USER, content="hi"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="tu",
                            tool_name=SKILL_LOAD_TOOL_NAME,
                            tool_kwargs={"name": "skill-creator"},
                        )
                    ]
                },
            ),
        ]
    )
    await interceptor.intercept(after)
    assert {t.name for t in after.state.input.context_stack.all_tools()} == {
        "echo",
        "present_files",
    }
