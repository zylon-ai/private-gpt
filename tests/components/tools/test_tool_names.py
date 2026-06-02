from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.tool_names import (
    WEB_FETCH_TOOL_NAME,
    resolve_internal_tool_name,
)


def test_resolve_internal_tool_name_accepts_web_extract_legacy_alias() -> None:
    assert resolve_internal_tool_name("web_extract_v1") == WEB_FETCH_TOOL_NAME


def test_tool_spec_original_name_canonicalizes_web_extract_legacy_alias() -> None:
    tool = ToolSpec(name="web_extract", type="web_extract_v1")
    assert tool.get_original_tool_name() == WEB_FETCH_TOOL_NAME
