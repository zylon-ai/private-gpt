from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ChatMessage

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.tools.builders.text_editor_code_execution_tool_builder import (
    TextEditorCodeExecutionToolBuilder,
)
from private_gpt.components.tools.events.adapters import (
    BashCodeExecutionEventAdapter,
    TextEditorCodeExecutionEventAdapter,
)
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptorContext,
    ToolExecutionRequest,
    execute_tool_request,
)
from private_gpt.components.tools.tool_execution_outcome import (
    ToolExecutionFailure,
    ToolExecutionSuccess,
)
from private_gpt.components.tools.utils import require_tool_params
from private_gpt.events.models import (
    BashCodeExecutionResultBlock,
    TextBlock,
)
from private_gpt.server.chat.interceptors.configure_tool_execution_interceptor import (
    ConfigureToolExecutionInterceptor,
)
from private_gpt.server.chat.interceptors.null_tool_values_interceptor import (
    NullToolValuesRequestInterceptor,
)
from private_gpt.server.chat.interceptors.required_tool_params_interceptor import (
    RequiredToolParamsInterceptor,
)
from private_gpt.server.chat.interceptors.schema_coercing_tool_interceptor import (
    SchemaCoercingToolInterceptor,
)


def _bash_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": ["integer", "null"]},
            "restart": {"type": "boolean"},
        },
        "required": ["command"],
    }


def _editor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "path": {"type": "string"},
            "file_text": {"type": ["string", "null"]},
            "insert_text": {"type": ["string", "null"]},
            "new_str": {"type": ["string", "null"]},
            "insert_line": {"type": ["integer", "null"]},
        },
        "required": ["command", "path"],
    }


async def _run_bash(command: str, timeout: int | None = None) -> list[Any]:
    return [
        BashCodeExecutionResultBlock(
            stdout=command,
            stderr="",
            return_code=0,
        )
    ]


def _bash_spec() -> ToolSpec:
    return ToolSpec.from_defaults(
        name="bash",
        type="bash_v1",
        runtime="server",
        event_adapter=BashCodeExecutionEventAdapter,
        input_schema=_bash_schema(),
        async_fn=_run_bash,
    )


def _editor_spec() -> ToolSpec:
    async def text_editor(command: str, path: str, **kwargs: Any) -> list[Any]:
        return [TextBlock(text=f"{command}:{path}")]

    return ToolSpec.from_defaults(
        name="text_editor_code_execution",
        type="text_editor_code_execution_v1",
        runtime="server",
        event_adapter=TextEditorCodeExecutionEventAdapter,
        input_schema=_editor_schema(),
        async_fn=text_editor,
    )


def _request(
    tool_spec: ToolSpec,
    tool_kwargs: dict[str, Any],
    *,
    tool_name: str | None = None,
) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        tool_id="tool-1",
        tool_name=tool_name or tool_spec.name or "tool",
        tool_kwargs=tool_kwargs,
        tool_spec=tool_spec,
    )


def _configure() -> ConfigureToolExecutionInterceptor:
    return ConfigureToolExecutionInterceptor(
        null_tool_values_interceptor=NullToolValuesRequestInterceptor(),
        schema_coercing_interceptor=SchemaCoercingToolInterceptor(),
        required_tool_params_interceptor=RequiredToolParamsInterceptor(),
    )


def test_require_tool_params_accepts_present_values() -> None:
    require_tool_params(
        "bash",
        {"command": "echo ok", "timeout": None},
        {"required": ["command"]},
    )


def test_require_tool_params_accepts_empty_string() -> None:
    require_tool_params("create", {"file_text": ""}, {"required": ["file_text"]})


@pytest.mark.parametrize(
    ("kwargs", "schema"),
    [
        ({}, {"required": ["command"]}),
        ({"command": None}, {"required": ["command"]}),
        ({"timeout": 5}, {"required": ["command"]}),
    ],
)
def test_require_tool_params_rejects_missing_or_none(
    kwargs: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"bash requires the command parameter\."):
        require_tool_params("bash", kwargs, schema)


def test_require_tool_params_skips_when_schema_has_no_required() -> None:
    require_tool_params("list_skills", {"page": 0}, {"properties": {}})
    require_tool_params("list_skills", {}, None)


@pytest.mark.anyio
async def test_required_interceptor_raises_after_nulls_are_stripped() -> None:
    interceptor = RequiredToolParamsInterceptor()
    context = ToolExecutionInterceptorContext(
        phase=InterceptorPhase.BEFORE_TOOL,
        request=_request(_bash_spec(), {}),
        tool_kwargs={"timeout": 5},
    )

    with pytest.raises(ValueError, match=r"bash requires the command parameter\."):
        await interceptor.intercept(context)


@pytest.mark.anyio
async def test_required_interceptor_ignores_after_tool_phase() -> None:
    interceptor = RequiredToolParamsInterceptor()
    context = ToolExecutionInterceptorContext(
        phase=InterceptorPhase.AFTER_TOOL,
        request=_request(_bash_spec(), {}),
        tool_kwargs={},
    )

    await interceptor.intercept(context)


@pytest.mark.anyio
async def test_configure_chain_strips_null_required_then_rejects() -> None:
    context = ToolExecutionInterceptorContext(
        phase=InterceptorPhase.BEFORE_TOOL,
        request=_request(_bash_spec(), {"command": None, "timeout": "10"}),
        tool_kwargs={"command": None, "timeout": "10"},
    )

    with pytest.raises(ValueError, match=r"bash requires the command parameter\."):
        await _configure().intercept(context)

    assert context.tool_kwargs == {"timeout": 10}


@pytest.mark.anyio
async def test_executor_returns_formatted_failure_for_missing_required_param() -> None:
    response = await execute_tool_request(
        _request(_bash_spec(), {"command": None}),
        interceptors=[_configure()],
    )

    assert response.is_error is True
    assert isinstance(response.outcome, ToolExecutionFailure)
    assert response.outcome.error.message == "bash requires the command parameter."
    assert response.outcome.error.exception_type == "ValueError"
    assert response.tool_message.role == "tool"
    assert response.tool_message.content == "bash requires the command parameter."
    assert isinstance(response.tool_message, ChatMessage)
    assert response.tool_message.additional_kwargs["tool_call_id"] == "tool-1"
    assert response.tool_message.additional_kwargs["tool_call_name"] == "bash"

    result = BashCodeExecutionEventAdapter().build_tool_result(
        tool_use_id="tool-1",
        outcome=response.outcome,
    )
    assert result.content.type == "bash_code_execution_tool_result_error"
    assert result.content.render() == "Error: bash requires the command parameter."


@pytest.mark.anyio
async def test_executor_returns_formatted_failure_for_missing_editor_path() -> None:
    response = await execute_tool_request(
        _request(_editor_spec(), {"command": "insert", "file_text": "oops"}),
        interceptors=[_configure()],
    )

    assert response.is_error is True
    assert isinstance(response.outcome, ToolExecutionFailure)
    assert (
        response.outcome.error.message
        == "text_editor_code_execution requires the path parameter."
    )
    assert response.tool_message.content == (
        "text_editor_code_execution requires the path parameter."
    )

    result = TextEditorCodeExecutionEventAdapter().build_tool_result(
        tool_use_id="tool-1",
        outcome=response.outcome,
    )
    assert result.content.type == "text_editor_code_execution_tool_result_error"
    assert result.content.render() == (
        "Error: text_editor_code_execution requires the path parameter."
    )


@pytest.mark.anyio
async def test_executor_runs_tool_when_required_params_are_present() -> None:
    response = await execute_tool_request(
        _request(_bash_spec(), {"command": "echo ok", "timeout": None}),
        interceptors=[_configure()],
    )

    assert response.is_error is False
    assert isinstance(response.outcome, ToolExecutionSuccess)
    assert response.outcome.content[0].stdout == "echo ok"
    assert response.tool_message.content == response.outcome.content[0].render()
    assert "echo ok" in (response.tool_message.content or "")


@pytest.mark.anyio
async def test_executor_accepts_empty_string_required_param() -> None:
    async def create(path: str, file_text: str) -> list[Any]:
        return [TextBlock(text=f"created {path} ({len(file_text)})")]

    spec = ToolSpec.from_defaults(
        name="create",
        type="create_v1",
        runtime="server",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "file_text": {"type": "string"},
            },
            "required": ["path", "file_text"],
        },
        async_fn=create,
    )
    response = await execute_tool_request(
        _request(spec, {"path": "empty.txt", "file_text": ""}),
        interceptors=[_configure()],
    )

    assert response.is_error is False
    assert isinstance(response.outcome, ToolExecutionSuccess)
    assert response.tool_message.content == "created empty.txt (0)"


def _child_tool(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=f"{name}_v1",
        async_fn=AsyncMock(return_value=[TextBlock(text=name)]),
    )


@pytest.mark.anyio
async def test_executor_formats_insert_when_llm_passes_create_param() -> None:
    child_builder = SimpleNamespace(
        build_view_tool=AsyncMock(return_value=_child_tool("view")),
        build_str_replace_tool=AsyncMock(return_value=_child_tool("str_replace")),
        build_create_tool=AsyncMock(return_value=_child_tool("create")),
        build_insert_tool=AsyncMock(return_value=_child_tool("insert")),
    )
    spec = await TextEditorCodeExecutionToolBuilder(child_builder).build_tool(
        SimpleNamespace(session_id="corr-editor", env={}, mounts=[])
    )
    response = await execute_tool_request(
        _request(
            spec,
            {
                "command": "insert",
                "path": "/home/agent/workspace/test_editor.md",
                "insert_line": 5,
                "file_text": "Line 6: This is a newly inserted line!",
            },
        ),
        interceptors=[_configure()],
    )

    assert response.is_error is True
    assert isinstance(response.outcome, ToolExecutionFailure)
    assert "insert requires the insert_text parameter" in response.outcome.error.message
    assert "splitlines" not in response.outcome.error.message
    assert response.tool_message.role == "tool"
    assert "insert requires the insert_text parameter" in (
        response.tool_message.content or ""
    )

    result = TextEditorCodeExecutionEventAdapter().build_tool_result(
        tool_use_id="tool-1",
        outcome=response.outcome,
    )
    assert result.content.type == "text_editor_code_execution_tool_result_error"
    assert "insert requires the insert_text parameter" in result.content.render()
    child_builder.build_insert_tool.return_value.async_fn.assert_not_awaited()
