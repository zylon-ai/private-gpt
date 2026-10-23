CODE_EXECUTION_ERROR_MESSAGES: dict[str, str] = {
    "unavailable": "Tool temporarily unavailable. Try again or use an alternative approach.",
    "execution_time_exceeded": "Execution timed out. Break the task into smaller steps or reduce the workload.",
    "invalid_tool_input": "Invalid tool input. Check the required parameters and try again.",
    "too_many_requests": "Rate limit reached. Wait before retrying.",
    "output_file_too_large": "Output too large. Limit output size (e.g. use head/tail, write to a file instead).",
    "file_not_found": "File not found. Check the path and make sure the file exists before retrying.",
}

WEB_SEARCH_ERROR_MESSAGES: dict[str, str] = {
    "invalid_tool_input": "Invalid search query. Revise the query and try again.",
    "unavailable": "Web search temporarily unavailable. Try again or rephrase the request.",
    "max_uses_exceeded": "Maximum number of web searches reached for this request.",
    "too_many_requests": "Rate limit exceeded. Wait before retrying.",
    "query_too_long": "Search query too long. Shorten it and try again.",
    "request_too_large": "Request too large, typically due to a long domain filter list.",
}

WEB_FETCH_ERROR_MESSAGES: dict[str, str] = {
    "invalid_tool_input": "Invalid URL or parameters. Check the URL and try again.",
    "url_too_long": "URL too long. Shorten the URL and try again.",
    "url_not_allowed": "URL not allowed by the current policy.",
    "url_not_in_prior_context": "URL was not referenced earlier in the conversation.",
    "url_not_accessible": "URL could not be reached. Check that it is publicly accessible.",
    "unsupported_content_type": "Content type not supported for fetching.",
    "too_many_requests": "Rate limit exceeded. Wait before retrying.",
    "max_uses_exceeded": "Maximum number of web fetches reached for this request.",
    "unavailable": "Web fetch temporarily unavailable. Try again or use an alternative approach.",
}
