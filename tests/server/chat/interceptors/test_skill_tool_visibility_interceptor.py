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
