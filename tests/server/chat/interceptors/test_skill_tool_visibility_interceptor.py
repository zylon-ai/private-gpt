import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
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
