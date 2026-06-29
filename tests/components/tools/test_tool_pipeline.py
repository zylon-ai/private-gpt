from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillFrontmatter,
    SkillVersionEntity,
)
from private_gpt.components.tools.processors.base import _session_id
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
from private_gpt.settings.settings import unsafe_typed_settings


def _request(tools: list[ToolSpec]) -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        tool_config=ResolvedToolConfig(tools=tools),
        context=ResolvedContextConfig(correlation_id="corr-123"),
    )


@pytest.mark.asyncio
async def test_tool_pipeline_recursively_expands_code_execution_wrapper() -> None:
    bash_builder = SimpleNamespace(
        build_tool=AsyncMock(
            side_effect=lambda session_id, name="bash", type="bash_v1", bundles=None, **kw: ToolSpec.from_defaults(
                name=name,
                type=type,
                description="bash",
                async_fn=AsyncMock(return_value=[]),
            )
        )
    )
    text_editor_builder = SimpleNamespace(
        build_view_tool=AsyncMock(
            side_effect=lambda session_id, name="view", type="view_v1", bundles=None, **kw: ToolSpec.from_defaults(
                name=name,
                type=type,
                description="view",
                async_fn=AsyncMock(return_value=[]),
            )
        ),
        build_str_replace_tool=AsyncMock(
            side_effect=lambda session_id, name="str_replace", type="str_replace_v1", bundles=None, **kw: ToolSpec.from_defaults(
                name=name,
                type=type,
                description="replace",
                async_fn=AsyncMock(return_value=[]),
            )
        ),
        build_create_tool=AsyncMock(
            side_effect=lambda session_id, name="create", type="create_v1", bundles=None, **kw: ToolSpec.from_defaults(
                name=name,
                type=type,
                description="create",
                async_fn=AsyncMock(return_value=[]),
            )
        ),
        build_insert_tool=AsyncMock(
            side_effect=lambda session_id, name="insert", type="insert_v1", bundles=None, **kw: ToolSpec.from_defaults(
                name=name,
                type=type,
                description="insert",
                async_fn=AsyncMock(return_value=[]),
            )
        ),
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
        bash_processor=BashProcessor(bash_builder),
        text_editor_processor=TextEditorProcessor(text_editor_builder),
        present_files_processor=noop,
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
        "bash",
        "view",
        "str_replace",
        "create",
        "insert",
    ]


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
    settings = unsafe_typed_settings.model_copy(deep=True)
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
