from typing import Any

from llama_index.core.tools import FunctionTool

from private_gpt.components.tools.tool_names import (
    BASH_TOOL_NAME,
    CODE_EXECUTION_TOOL_NAME,
    DATABASE_QUERY_TOOL_NAME,
    PRESENT_FILES_TOOL_NAME,
    PRESENT_SERVER_TOOL_NAME,
    SEMANTIC_SEARCH_TOOL_NAME,
    SKILLS_TOOL_NAME,
    SUMMARIZE_TOOL_NAME,
    TABULAR_DATA_ANALYSIS,
    TEXT_EDITOR_CREATE_TOOL_NAME,
    TEXT_EDITOR_INSERT_TOOL_NAME,
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    TEXT_EDITOR_TOOL_NAME,
    TEXT_EDITOR_VIEW_TOOL_NAME,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
)


def _placeholder_fn(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("This is a placeholder function for a internal tool.")


def _placeholder_tool(name: str, description: str) -> FunctionTool:
    return FunctionTool.from_defaults(
        name=name,
        description=description,
        fn=_placeholder_fn,
        return_direct=True,
    )


SEMANTIC_SEARCH_TOOL_FN = _placeholder_tool(
    SEMANTIC_SEARCH_TOOL_NAME,
    "Perform semantic search over the files in the knowledge base. "
    "Searches should only be done for a single topic at a time, so this tool should be "
    "used multiple times to get better results.",
)

TABULAR_DATA_TOOL_FN = _placeholder_tool(
    TABULAR_DATA_ANALYSIS,
    "Perform Pandas queries over the files in the knowledge base. "
    "It receives a query where it explain what to do, it will perform a search to find all tables"
    "and it will run python code to generate aggregate queries, sorters, charts, etc.",
)

DATABASE_QUERY_TOOL_FN = _placeholder_tool(
    DATABASE_QUERY_TOOL_NAME,
    "Run a search using natural language against connected databases and return the results.",
)

SUMMARIZE_TOOL_FN = _placeholder_tool(
    SUMMARIZE_TOOL_NAME,
    "Summarize the content of the files in the knowledge base. "
    "Summarization should be used to get a high level overview of the content of "
    "the knowledge base, and should be used multiple times to get better results.",
)

WEB_FETCH_TOOL_FN = _placeholder_tool(
    WEB_FETCH_TOOL_NAME,
    "Extract text from a web by it's url."
    "It will return the text content of the page in human readable format."
    "This tool should only be used when the content of a website is needed, "
    "not for general web search. The website url must use http or https protocol.",
)

WEB_SEARCH_TOOL_FN = _placeholder_tool(
    WEB_SEARCH_TOOL_NAME,
    "Search the web for relevant information."
    "It will return a list of relevant links and a brief summary of each link."
    "This tool should only be used when the content of a website is needed, "
    "not for general web search. The website url must use http or https protocol.",
)

CODE_EXECUTION_TOOL_FN = _placeholder_tool(
    CODE_EXECUTION_TOOL_NAME,
    "Execute commands and manipulate files in the session workspace.",
)

BASH_TOOL_FN = _placeholder_tool(
    BASH_TOOL_NAME,
    "Execute bash commands in the session workspace.",
)

TEXT_EDITOR_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_TOOL_NAME,
    "View and edit files in the session workspace.",
)

TEXT_EDITOR_VIEW_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_VIEW_TOOL_NAME,
    "View a file or directory in the session workspace.",
)

TEXT_EDITOR_STR_REPLACE_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    "Replace a single exact string in a file.",
)

TEXT_EDITOR_CREATE_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_CREATE_TOOL_NAME,
    "Create a new file in the session workspace.",
)

TEXT_EDITOR_INSERT_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_INSERT_TOOL_NAME,
    "Insert text into a file after a given line number.",
)

PRESENT_FILES_TOOL_FN = _placeholder_tool(
    PRESENT_FILES_TOOL_NAME,
    "Present one or more output files to the user after they have been created.",
)

PRESENT_SERVER_TOOL_FN = _placeholder_tool(
    PRESENT_SERVER_TOOL_NAME,
    (
        "Expose an HTTP service running inside the sandbox on a given port and present "
        "its URL to the user. Call this after starting a server (e.g. a web app, "
        "Jupyter, Streamlit) so the user can open or interact with it."
    ),
)

SKILLS_TOOL_FN = _placeholder_tool(
    SKILLS_TOOL_NAME,
    "Load, unload, and list skills available to the current conversation.",
)
