from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedToolConfig,
    ToolExecutionMetadata,
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillFrontmatter,
    SkillVersionEntity,
)
from private_gpt.components.tools.processors.anthropic_tool_translation_processor import (
    AnthropicToolTranslationProcessor,
)
from private_gpt.components.tools.processors.base import _replace_tool, _session_id
from private_gpt.components.tools.processors.bash_processor import BashProcessor
from private_gpt.components.tools.processors.code_execution_processor import (
    CodeExecutionProcessor,
)
from private_gpt.components.tools.processors.skill_management_processor import (
    SkillManagementProcessor,
)
from private_gpt.components.tools.processors.text_editor_processor import (
    TextEditorProcessor,
)
from private_gpt.components.tools.tool_pipeline import ToolPipeline
from private_gpt.server.utils.artifact_input import SkillArtifact
from private_gpt.settings.settings import settings as _load_settings


def _request(tools: list[ToolSpec]) -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        tool_config=ResolvedToolConfig(tools=tools),
        context=ResolvedContextConfig(correlation_id="corr-123"),
    )


def test_replace_tool_preserves_single_replacement_properties() -> None:
    original = ToolSpec(
        name="semantic_search",
        type="semantic_search_v1",
        description="Custom search description",
        defer_loading=True,
        partial_params={"scope": "project"},
        instructions="Use the project knowledge base.",
        requirements=[ToolRequirements.SANDBOX],
    )
    replacement = ToolSpec.from_defaults(
        name="semantic_search",
        type="semantic_search_v1",
        runtime="server",
        description="Default search description",
        async_fn=AsyncMock(return_value=[]),
    )
    request = _request([original])

    assert _replace_tool(request, original, [replacement])

    resolved = request.tool_config.tools[0]
    assert resolved.description == "Custom search description"
    assert resolved.defer_loading is True
    assert resolved.partial_params == {"scope": "project"}
    assert resolved.instructions == "Use the project knowledge base."
    assert resolved.requirements == [ToolRequirements.SANDBOX]
    assert resolved.runtime == "server"
    assert resolved.async_fn is replacement.async_fn


def test_replace_tool_preserves_shared_properties_across_expansion() -> None:
    original = ToolSpec(
        name="code_execution",
        type="code_execution_v1",
        description="Wrapper description",
        defer_loading=True,
        partial_params={"unsafe_for_children": True},
        instructions="Use the shared sandbox carefully.",
        requirements=[ToolRequirements.SANDBOX],
    )
    replacements = [
        ToolSpec.from_defaults(
            name="bash",
            type="bash_v1",
            runtime="server",
            description="Bash description",
            async_fn=AsyncMock(return_value=[]),
        ),
        ToolSpec.from_defaults(
            name="text_editor",
            type="text_editor_v1",
            runtime="server",
            description="Editor description",
            async_fn=AsyncMock(return_value=[]),
        ),
    ]
    request = _request([original])

    assert _replace_tool(request, original, replacements)

    bash, editor = request.tool_config.tools
    assert bash.description == "Bash description"
    assert editor.description == "Editor description"
    assert bash.partial_params is None
    assert editor.partial_params is None
    assert all(tool.defer_loading for tool in (bash, editor))
    assert all(
        tool.instructions == "Use the shared sandbox carefully."
        for tool in (bash, editor)
    )
    assert all(
        tool.requirements == [ToolRequirements.SANDBOX] for tool in (bash, editor)
    )


@pytest.mark.asyncio
async def test_tool_pipeline_recursively_expands_code_execution_wrapper() -> None:
    bash_builder = SimpleNamespace(
        build_tool=AsyncMock(
            side_effect=lambda session_id, name="bash_code_execution", type="bash_code_execution_v1", **kw: (
                ToolSpec.from_defaults(
                    name=name,
                    type=type,
                    description="bash",
                    async_fn=AsyncMock(return_value=[]),
                )
            )
        )
    )
    unified_text_editor_builder = SimpleNamespace(
        build_tool=AsyncMock(
            side_effect=lambda session_id, name="text_editor_code_execution", type="text_editor_code_execution_v1", **kw: (
                ToolSpec.from_defaults(
                    name=name,
                    type=type,
                    description="text_editor",
                    async_fn=AsyncMock(return_value=[]),
                )
            )
        )
    )
    text_editor_child_builder = SimpleNamespace(
        build_view_tool=AsyncMock(return_value=None),
        build_str_replace_tool=AsyncMock(return_value=None),
        build_create_tool=AsyncMock(return_value=None),
        build_insert_tool=AsyncMock(return_value=None),
    )
    noop = SimpleNamespace(intercept=AsyncMock(return_value=False))
    pipeline = ToolPipeline(
        anthropic_tool_translation_processor=noop,
        semantic_search_processor=noop,
        tabular_data_processor=noop,
        database_query_processor=noop,
        web_fetch_processor=noop,
        web_search_processor=noop,
        skill_management_processor=noop,
        code_execution_processor=CodeExecutionProcessor(),
        bash_processor=BashProcessor(bash_builder, _settings()),
        text_editor_processor=TextEditorProcessor(
            text_editor_child_builder, unified_text_editor_builder
        ),
        present_files_processor=noop,
        present_server_processor=noop,
    )
    request = _request(
        [
            ToolSpec(
                name="code_execution",
                type="code_execution_v1",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )

    resolved = await pipeline.contextualize_internal_tools(request)

    assert [tool.name for tool in resolved.tool_config.tools] == [
        "bash_code_execution",
        "text_editor_code_execution",
    ]


_DUMMY_METADATA = ToolExecutionMetadata(
    rebuild_callable="private_gpt.components.tools.builders.text_editor_tool_builder:rebuild_text_editor_create_tool",
    rebuild_kwargs={},
)


def _built_tool(name: str, tool_type: str) -> ToolSpec:
    """Return a fully-built ToolSpec (async_fn + execution_metadata set)."""
    return ToolSpec.from_defaults(
        name=name,
        type=tool_type,
        description=name,
        async_fn=AsyncMock(return_value=[]),
        execution_metadata=_DUMMY_METADATA,
    )


def _make_bash_builder() -> SimpleNamespace:
    return SimpleNamespace(
        build_tool=AsyncMock(
            return_value=_built_tool("bash_code_execution", "bash_code_execution_v1")
        ),
    )


def _make_text_editor_child_builder() -> SimpleNamespace:
    return SimpleNamespace(
        build_view_tool=AsyncMock(return_value=_built_tool("view", "view_v1")),
        build_str_replace_tool=AsyncMock(
            return_value=_built_tool("str_replace", "str_replace_v1")
        ),
        build_create_tool=AsyncMock(return_value=_built_tool("create", "create_v1")),
        build_insert_tool=AsyncMock(return_value=_built_tool("insert", "insert_v1")),
    )


def _make_text_editor_builder() -> SimpleNamespace:
    return SimpleNamespace(
        build_tool=AsyncMock(
            return_value=_built_tool(
                "text_editor_code_execution", "text_editor_code_execution_v1"
            )
        ),
    )


def _make_skill_processor() -> SkillManagementProcessor:
    return SkillManagementProcessor(
        settings=_settings(),
        skill_service=SimpleNamespace(
            recover_versions=AsyncMock(return_value=[_skill_version()])
        ),
    )


def _make_pipeline(
    *,
    anthropic: bool = False,
    skill_processor: SkillManagementProcessor | None = None,
) -> ToolPipeline:
    noop = SimpleNamespace(intercept=AsyncMock(return_value=False))
    return ToolPipeline(
        anthropic_tool_translation_processor=(
            AnthropicToolTranslationProcessor() if anthropic else noop
        ),
        semantic_search_processor=noop,
        tabular_data_processor=noop,
        database_query_processor=noop,
        web_fetch_processor=noop,
        web_search_processor=noop,
        skill_management_processor=skill_processor or noop,
        code_execution_processor=CodeExecutionProcessor(),
        bash_processor=BashProcessor(_make_bash_builder(), _settings()),
        text_editor_processor=TextEditorProcessor(
            _make_text_editor_child_builder(), _make_text_editor_builder()
        ),
        present_files_processor=noop,
        present_server_processor=noop,
    )


_SKILL_ARTIFACT = SkillArtifact(
    skill_filter=SkillFilter(
        collection="tenant-a",
        skill_or_version_ids=["skill_1"],
    )
)

_NESTED_EXPANSION_CASES = [
    pytest.param(
        # text_editor_v1 → TextEditorProcessor → build text_editor_code_execution
        lambda: _make_pipeline(),
        ToolSpec(
            name="text_editor",
            type="text_editor_v1",
            input_schema={"type": "object", "properties": {}},
        ),
        None,
        ["text_editor_code_execution"],
        id="text_editor_expands_to_built_leaf_tools",
    ),
    pytest.param(
        # code_execution_v1 → [bash_code_execution, text_editor_code_execution]
        #   bash_code_execution     → BashProcessor builds it
        #   text_editor_code_execution → TextEditorProcessor builds it
        lambda: _make_pipeline(),
        ToolSpec(
            name="code_execution",
            type="code_execution_v1",
            input_schema={"type": "object", "properties": {}},
        ),
        None,
        ["bash_code_execution", "text_editor_code_execution"],
        id="code_execution_fully_expands_all_levels",
    ),
    pytest.param(
        # code_execution_20250825 (Anthropic wire type) → translate → code_execution_v1
        #   … same tree as above
        lambda: _make_pipeline(anthropic=True),
        ToolSpec(
            name="code_execution",
            type="code_execution_20250825",
            input_schema={"type": "object", "properties": {}},
        ),
        None,
        ["bash_code_execution", "text_editor_code_execution"],
        id="code_execution_anthropic_wire_type_fully_expands",
    ),
    pytest.param(
        # skills_v1 → expand → [load_skill, unload_skill, list_skills] stubs
        #           → build  → 3 built leaf tools
        lambda: _make_pipeline(skill_processor=_make_skill_processor()),
        ToolSpec(
            name="skills",
            type="skills_v1",
            input_schema={"type": "object", "properties": {}},
        ),
        _SKILL_ARTIFACT,
        ["load_skill", "unload_skill", "list_skills"],
        id="skills_expands_to_built_leaf_tools",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pipeline_factory", "input_tool", "tool_artifact", "expected_names"),
    _NESTED_EXPANSION_CASES,
)
async def test_nested_tool_expansion_fully_builds_all_leaf_tools(
    pipeline_factory: object,
    input_tool: ToolSpec,
    tool_artifact: SkillArtifact | None,
    expected_names: list[str],
) -> None:
    """Every leaf tool produced by nested expand→build chains must be fully built.

    Regression: the pipeline ran each processor once, so parent stubs expanded
    to child stubs that were never fed back through the processors that build them.
    Child stubs had execution_metadata=None, causing the LLM to receive an
    unresolved tool (e.g. 'text_editor_code_execution') and respond with
    "Tool not found." at runtime.
    """
    pipeline = pipeline_factory()
    request = _request([input_tool])
    if tool_artifact is not None:
        request.tool_context = [tool_artifact]

    resolved = await pipeline.contextualize_internal_tools(request)

    assert [t.name for t in resolved.tool_config.tools] == expected_names
    for tool in resolved.tool_config.tools:
        assert tool.execution_metadata is not None, (
            f"Tool {tool.name!r} (type={tool.type!r}) is missing execution_metadata "
            "— it was left as an unbuilt stub"
        )


def test_tool_pipeline_uses_user_id_as_session_id() -> None:
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        tool_config=ResolvedToolConfig(tools=[]),
        context=ResolvedContextConfig(
            user_id="session-123",
            correlation_id="corr-123",
        ),
    )

    assert _session_id(request) == "session-123"


def _skill_version() -> SkillVersionEntity:
    return SkillVersionEntity(
        id="skillver_1",
        skill_id="skill_1",
        version="1000000",
        frontmatter=SkillFrontmatter(name="my-skill", description="Test skill"),
        storage_prefix="skills/tenant-a/skill_1/1000000",
        created_at=datetime.now(tz=UTC),
    )


def _settings():
    settings = _load_settings().model_copy(deep=True)
    settings.skills.skill_injection_mode = "system_prompt"
    return settings


@pytest.mark.asyncio
async def test_skill_tools_are_built_without_pre_recovery() -> None:
    recover = AsyncMock(return_value=[_skill_version()])
    noop = SimpleNamespace(intercept=AsyncMock(return_value=False))
    pipeline = ToolPipeline(
        anthropic_tool_translation_processor=noop,
        semantic_search_processor=noop,
        tabular_data_processor=noop,
        database_query_processor=noop,
        web_fetch_processor=noop,
        web_search_processor=noop,
        skill_management_processor=SkillManagementProcessor(
            settings=_settings(),
            skill_service=SimpleNamespace(recover_versions=recover),
        ),
        code_execution_processor=CodeExecutionProcessor(),
        bash_processor=noop,
        text_editor_processor=noop,
        present_files_processor=noop,
        present_server_processor=noop,
    )
    request = _request(
        [
            ToolSpec(
                name="load_skill",
                type="load_skill_v1",
                input_schema={"type": "object", "properties": {}},
            ),
            ToolSpec(
                name="load_skill",
                type="load_skill_v1",
                input_schema={"type": "object", "properties": {}},
            ),
            ToolSpec(
                name="list_skills",
                type="list_skills_v1",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
    )
    request.tool_context = [
        SkillArtifact(
            skill_filter=SkillFilter(
                collection="tenant-a",
                skill_or_version_ids=["skill_1"],
            )
        )
    ]

    resolved = await pipeline.contextualize_internal_tools(request)

    assert recover.await_count == 0
    assert len(resolved.tool_config.tools) == 3
    assert [tool.type for tool in resolved.tool_config.tools] == [
        "load_skill_v1",
        "load_skill_v1",
        "list_skills_v1",
    ]


@pytest.mark.asyncio
async def test_tool_pipeline_expands_skills_wrapper() -> None:
    recover = AsyncMock(return_value=[_skill_version()])
    noop = SimpleNamespace(intercept=AsyncMock(return_value=False))
    pipeline = ToolPipeline(
        anthropic_tool_translation_processor=noop,
        semantic_search_processor=noop,
        tabular_data_processor=noop,
        database_query_processor=noop,
        web_fetch_processor=noop,
        web_search_processor=noop,
        skill_management_processor=SkillManagementProcessor(
            settings=_settings(),
            skill_service=SimpleNamespace(recover_versions=recover),
        ),
        code_execution_processor=CodeExecutionProcessor(),
        bash_processor=noop,
        text_editor_processor=noop,
        present_files_processor=noop,
        present_server_processor=noop,
    )
    request = _request(
        [
            ToolSpec(
                name="skills",
                type="skills_v1",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    request.tool_context = [
        SkillArtifact(
            skill_filter=SkillFilter(
                collection="tenant-a",
                skill_or_version_ids=["skill_1"],
            )
        )
    ]

    resolved = await pipeline.contextualize_internal_tools(request)

    assert recover.await_count == 0
    assert [tool.name for tool in resolved.tool_config.tools] == [
        "load_skill",
        "unload_skill",
        "list_skills",
    ]
