import re

# General tools
SEMANTIC_SEARCH_TOOL_NAME = "semantic_search"
TABULAR_DATA_ANALYSIS = "tabular_analysis"
SUMMARIZE_TOOL_NAME = "summarize"
DATABASE_QUERY_TOOL_NAME = "database_query"
WEB_FETCH_TOOL_NAME = "web_fetch"
WEB_FETCH_LEGACY_TOOL_NAMES = ["web_extract"]
WEB_SEARCH_TOOL_NAME = "web_search"

# Code execution tools
CODE_EXECUTION_TOOL_NAME = "code_execution"
BASH_TOOL_NAME = "bash"
TEXT_EDITOR_TOOL_NAME = "text_editor"
TEXT_EDITOR_VIEW_TOOL_NAME = "view"
TEXT_EDITOR_STR_REPLACE_TOOL_NAME = "str_replace"
TEXT_EDITOR_CREATE_TOOL_NAME = "create"
TEXT_EDITOR_INSERT_TOOL_NAME = "insert"

# Skill management tools
SKILLS_TOOL_NAME = "skills"
SKILL_LOAD_TOOL_NAME = "load_skill"
SKILL_UNLOAD_TOOL_NAME = "unload_skill"
SKILL_LIST_TOOL_NAME = "list_skills"

GENERAL_INTERNAL_TOOLS = [
    SEMANTIC_SEARCH_TOOL_NAME,
    TABULAR_DATA_ANALYSIS,
    SUMMARIZE_TOOL_NAME,
    DATABASE_QUERY_TOOL_NAME,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
]

CODE_EXECUTION_INTERNAL_TOOLS = [
    CODE_EXECUTION_TOOL_NAME,
    BASH_TOOL_NAME,
    TEXT_EDITOR_TOOL_NAME,
    TEXT_EDITOR_VIEW_TOOL_NAME,
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    TEXT_EDITOR_CREATE_TOOL_NAME,
    TEXT_EDITOR_INSERT_TOOL_NAME,
]

SKILL_MANAGEMENT_TOOLS = [
    SKILLS_TOOL_NAME,
    SKILL_LOAD_TOOL_NAME,
    SKILL_UNLOAD_TOOL_NAME,
    SKILL_LIST_TOOL_NAME,
]

INTERNAL_TOOLS = [
    *GENERAL_INTERNAL_TOOLS,
    *CODE_EXECUTION_INTERNAL_TOOLS,
    *SKILL_MANAGEMENT_TOOLS,
]


def resolve_internal_tool_name(tool_type: str | None) -> str | None:
    if not tool_type:
        return None

    normalized_tool_type = re.sub(r"_v\d+$", "", tool_type)
    if normalized_tool_type in WEB_FETCH_LEGACY_TOOL_NAMES:
        return WEB_FETCH_TOOL_NAME
    if normalized_tool_type in INTERNAL_TOOLS:
        return normalized_tool_type

    return None
