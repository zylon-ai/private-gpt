from private_gpt.utils.dependencies import format_missing_dependency_message


def test_format_missing_dependency_message_uses_inexact_for_single_extra() -> None:
    message = format_missing_dependency_message("MCP tools", extras="tool-mcp")

    assert (
        message == "MCP tools dependencies are not installed. "
        "Install with `uv sync --inexact --extra tool-mcp`."
    )


def test_format_missing_dependency_message_uses_inexact_for_multiple_extras() -> None:
    message = format_missing_dependency_message(
        "Database query",
        extras=("database-postgres", "database"),
    )

    assert message == (
        "Database query dependencies are not installed. Install with one of: "
        "`uv sync --inexact --extra database-postgres` or "
        "`uv sync --inexact --extra database`."
    )


def test_format_missing_dependency_message_can_disable_inexact_sync() -> None:
    message = format_missing_dependency_message(
        "MCP tools",
        extras="tool-mcp",
        keep_other_deps=False,
    )

    assert (
        message == "MCP tools dependencies are not installed. "
        "Install with `uv sync --extra tool-mcp`."
    )
