from typing import Any

from llama_index.core.tools import FunctionTool

from private_gpt.components.tools.tool_names import (
    BASH_CODE_EXECUTION_TOOL_NAME,
    BASH_TOOL_NAME,
    CODE_EXECUTION_TOOL_NAME,
    DATABASE_QUERY_TOOL_NAME,
    PRESENT_FILES_TOOL_NAME,
    PRESENT_SERVER_TOOL_NAME,
    SEMANTIC_SEARCH_TOOL_NAME,
    SKILLS_TOOL_NAME,
    SUMMARIZE_TOOL_NAME,
    TABULAR_DATA_ANALYSIS,
    TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME,
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
    "Search the knowledge base and return the most relevant fragments with attribution. "
    "Keep each query focused on a single topic — decompose multi-faceted questions into "
    "separate calls and refine successive queries as results reveal new aspects or documents. "
    "Optionally restrict scope to specific source documents with the `artifacts` parameter. "
    "Always provide a `query`; never call this tool without one.",
)

TABULAR_DATA_TOOL_FN = _placeholder_tool(
    TABULAR_DATA_ANALYSIS,
    "Analyze CSV tables from the knowledge base by running Pandas operations on them "
    "(aggregations, groupings, rankings, statistics, filtering, joins). "
    "Describe the desired operation in natural language, naming the columns and any "
    "grouping or filter conditions. CSV files only — this tool fails on PDF, DOCX, "
    "XLSX, TXT, and image files.",
)

DATABASE_QUERY_TOOL_FN = _placeholder_tool(
    DATABASE_QUERY_TOOL_NAME,
    "Query the live connected databases (e.g. PostgreSQL, MySQL) in natural language and "
    "return the results. Read-only — it never modifies data. Results include the generated "
    "SQL, the row count, and a CSV attachment when rows are returned. For static CSV files "
    "in the knowledge base, use the tabular analysis tool instead.",
)

SUMMARIZE_TOOL_FN = _placeholder_tool(
    SUMMARIZE_TOOL_NAME,
    "Produce a high-level overview or summary of content from the knowledge base. "
    "Use it to understand the overall theme and scope of the documents, and call it "
    "multiple times with different angles to build a complete picture.",
)

WEB_FETCH_TOOL_FN = _placeholder_tool(
    WEB_FETCH_TOOL_NAME,
    "Fetch a single web page by its URL and return the content in human-readable markdown. "
    "Use when the actual content of a specific page is needed (a URL the user provided, or "
    "a link found by web_search). The URL must use the http or https protocol. "
    "Not for general web search — use the web_search tool for that.",
)

WEB_SEARCH_TOOL_FN = _placeholder_tool(
    WEB_SEARCH_TOOL_NAME,
    "Search the web and return a ranked list of relevant links with a brief summary of each. "
    "Use for current events, news, recent data, and general information not covered by the "
    "knowledge base. Backed by a third-party service with strict rate limits — issue one call "
    "at a time and stop as soon as you have enough to answer.",
)

CODE_EXECUTION_TOOL_FN = _placeholder_tool(
    CODE_EXECUTION_TOOL_NAME,
    "Execute shell commands and manipulate files in the session workspace.",
)

BASH_TOOL_FN = _placeholder_tool(
    BASH_TOOL_NAME,
    "Execute a bash command in the session sandbox and return its stdout, stderr, and exit "
    "code. All commands share the same sandbox: files you create and packages you install "
    "persist across calls. Only the shell process is fresh — exported variables and the "
    "working directory do not persist between calls, so set what you need within each "
    "command. Batch as many independent operations as possible into a single call to "
    "improve performance, and for independent parallel tasks issue multiple tool calls in "
    "the same turn rather than one at a time. Pass `restart=true` to wipe and recreate the "
    "workspace. Output is truncated to the configured maximum — keep commands focused and "
    "inspect failures before moving on.",
)

BASH_CODE_EXECUTION_TOOL_FN = _placeholder_tool(
    BASH_CODE_EXECUTION_TOOL_NAME,
    "Execute a bash command in the session sandbox and return its stdout, stderr, and exit "
    "code. All commands share the same sandbox: files you create and packages you install "
    "persist across calls. Only the shell process is fresh — exported variables and the "
    "working directory do not persist between calls, so set what you need within each "
    "command. Batch as many independent operations as possible into a single call to "
    "improve performance, and for independent parallel tasks issue multiple tool calls in "
    "the same turn rather than one at a time. Pass `restart=true` to wipe and recreate the "
    "workspace. Output is truncated to the configured maximum — keep commands focused and "
    "inspect failures before moving on.",
)

TEXT_EDITOR_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_TOOL_NAME,
    "View and edit files in the session workspace: read files and directories, create or "
    "overwrite files, and apply precise string replacements or line insertions.",
)

TEXT_EDITOR_CODE_EXECUTION_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME,
    "View and edit files in the session workspace. Dispatch with `command` set to one of: "
    "`view` (read a file, or list a directory), "
    "`create` (write or overwrite a file with `file_text`), "
    "`str_replace` (replace a single exact `old_str` with `new_str`; the match must be unique), "
    "`insert` (insert text after line `insert_line`). Always pass an absolute `path`. "
    "When viewing a file, always pass `view_range=[start, end]` — 1-based and inclusive, "
    "e.g. `[11, 20]` shows lines 11–20 and `end=-1` shows from `start` to the end of the "
    "file. Never view a whole large file in one call: the output is truncated at a fixed "
    "limit, and the result reports the visible line range plus the file's total line count, "
    "so page through large files window by window.",
)

TEXT_EDITOR_VIEW_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_VIEW_TOOL_NAME,
    "View a file or list a directory in the session workspace. "
    "When viewing a file, always pass `view_range=[start, end]` — 1-based and inclusive, "
    "e.g. `[11, 20]` shows lines 11–20 and `end=-1` shows from `start` to the end of the "
    "file. Never view a whole large file in one call: output is truncated at a fixed limit, "
    "and the result reports the visible line range plus the file's total line count, so page "
    "through large files window by window.",
)

TEXT_EDITOR_STR_REPLACE_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    "Replace a single exact string in a file. `old_str` must match exactly one occurrence "
    "(whitespace included) — if it appears more than once or not at all, the replacement "
    "fails. Include enough surrounding context to make the match unique.",
)

TEXT_EDITOR_CREATE_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_CREATE_TOOL_NAME,
    "Create or overwrite a file in the session workspace with the given `file_text`.",
)

TEXT_EDITOR_INSERT_TOOL_FN = _placeholder_tool(
    TEXT_EDITOR_INSERT_TOOL_NAME,
    "Insert text into a file after a given line number (`insert_line`; 0 inserts at the "
    "beginning of the file).",
)

PRESENT_FILES_TOOL_FN = _placeholder_tool(
    PRESENT_FILES_TOOL_NAME,
    "REQUIRED to show files to the user. Present one or more files that already exist under "
    "/mnt/user-data/outputs/ so they appear as downloadable attachments in the chat. Only "
    "outputs paths are accepted — copy workspace/uploads/skills files into "
    "/mnt/user-data/outputs/ first (e.g. `cp ... /mnt/user-data/outputs/`). Writing a file "
    "is not enough — if you do not call this tool, the user will never see the file.",
)

PRESENT_SERVER_TOOL_FN = _placeholder_tool(
    PRESENT_SERVER_TOOL_NAME,
    (
        "Expose an HTTP service running inside the sandbox on a given `port` and present "
        "its URL to the user. Call this after starting a server (e.g. a web app, Jupyter, "
        "Streamlit) so the user can open or interact with it. Optionally pass `service_name` "
        "and an `initial_path` to deep-link to a specific route."
    ),
)

SKILLS_TOOL_FN = _placeholder_tool(
    SKILLS_TOOL_NAME,
    "Manage skills for this conversation: list the available skill catalog (paginated), "
    "load a skill to inject its instructions into context, and unload a skill when it is "
    "no longer needed.",
)
