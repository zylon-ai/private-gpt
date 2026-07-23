import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.chat.input_models import BlobVisibilityMode
from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedSystemConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.sandbox.content_bundle import ContentBundle
from private_gpt.components.tools.builders.bash_tool_builder import BashToolBuilder
from private_gpt.components.tools.builders.database_query_builder import (
    DatabaseQueryToolBuilder,
)
from private_gpt.components.tools.builders.present_files_tool_builder import (
    PresentFilesToolBuilder,
)
from private_gpt.components.tools.builders.present_server_tool_builder import (
    PresentServerToolBuilder,
)
from private_gpt.components.tools.builders.semantic_search_builder import (
    SemanticSearchToolBuilder,
)
from private_gpt.components.tools.builders.tabular_data_builder import (
    TabularDataToolBuilder,
)
from private_gpt.components.tools.builders.text_editor_tool_builder import (
    TextEditorToolBuilder,
)
from private_gpt.components.tools.builders.web_fetch_builder import WebFetchToolBuilder
from private_gpt.components.tools.builders.web_search_builder import (
    WebSearchToolBuilder,
)
from private_gpt.components.tools.processors.bash_processor import BashProcessor
from private_gpt.components.tools.processors.database_query_processor import (
    DatabaseQueryProcessor,
)
from private_gpt.components.tools.processors.present_files_processor import (
    PresentFilesProcessor,
)
from private_gpt.components.tools.processors.present_server_processor import (
    PresentServerProcessor,
)
from private_gpt.components.tools.processors.semantic_search_processor import (
    SemanticSearchProcessor,
)
from private_gpt.components.tools.processors.tabular_data_processor import (
    TabularDataProcessor,
)
from private_gpt.components.tools.processors.text_editor_processor import (
    TextEditorProcessor,
)
from private_gpt.components.tools.processors.web_fetch_processor import (
    WebFetchProcessor,
)
from private_gpt.components.tools.processors.web_search_processor import (
    WebSearchProcessor,
)
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.server.utils.artifact_input import (
    IngestedArtifact,
    SqlDatabaseArtifact,
)


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, type=f"{name}_v1")


def _resolved(name: str) -> ToolSpec:
    return ToolSpec.from_defaults(
        name=name,
        type=f"{name}_v1",
        runtime="server",
        async_fn=AsyncMock(return_value=[]),
    )


def _request(
    tool: ToolSpec,
    *,
    tool_context: list[object] | None = None,
    content_bundles: list[ContentBundle] | None = None,
    bundles_to_remove: list[str] | None = None,
) -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(
            model="contract-model",
            prompt="Contract system prompt",
            blob_visibility=BlobVisibilityMode.INTERNAL,
        ),
        tool_config=ResolvedToolConfig(
            tools=[tool],
            validation_mode=ToolValidationMode.EAGER,
        ),
        tool_context=tool_context or [],
        context=ResolvedContextConfig(
            correlation_id="contract-correlation",
            maximum_context_length=98_765,
            content_bundles=content_bundles or [],
            bundles_to_remove=bundles_to_remove or [],
        ),
    )


@pytest.mark.parametrize(
    ("builder_method", "expected_parameters"),
    [
        (
            SemanticSearchToolBuilder.build_tool,
            {
                "context_filter",
                "model_id",
                "embed_model_id",
                "name",
                "type",
                "description",
                "validate",
                "runtime",
                "kwargs",
            },
        ),
        (
            TabularDataToolBuilder.build_tool,
            {
                "context_filter",
                "model_id",
                "embed_model_id",
                "llm",
                "name",
                "type",
                "description",
                "validate",
                "runtime",
                "blob_visibility",
                "kwargs",
            },
        ),
        (
            DatabaseQueryToolBuilder.build_tool,
            {
                "sql_artifacts",
                "chat_history",
                "name",
                "type",
                "description",
                "validate",
                "runtime",
                "blob_visibility",
            },
        ),
        (
            WebSearchToolBuilder.build_tool,
            {"model_id", "name", "type", "description", "validate", "runtime"},
        ),
        (
            WebFetchToolBuilder.build_tool,
            {"name", "type", "description", "runtime"},
        ),
        (
            BashToolBuilder.build_tool,
            {"config", "name", "type", "description"},
        ),
        (
            TextEditorToolBuilder.build_view_tool,
            {"config", "name", "type", "description"},
        ),
        (
            TextEditorToolBuilder.build_str_replace_tool,
            {"config", "name", "type", "description"},
        ),
        (
            TextEditorToolBuilder.build_create_tool,
            {"config", "name", "type", "description"},
        ),
        (
            TextEditorToolBuilder.build_insert_tool,
            {"config", "name", "type", "description"},
        ),
        (
            PresentFilesToolBuilder.build_tool,
            {"session_id", "bundles", "name", "type", "description"},
        ),
        (
            PresentServerToolBuilder.build_tool,
            {"session_id", "name", "type", "description"},
        ),
    ],
)
def test_processor_builder_contract_tracks_signature_changes(
    builder_method: object,
    expected_parameters: set[str],
) -> None:
    parameters = set(inspect.signature(builder_method).parameters) - {"self"}
    assert parameters == expected_parameters


@pytest.mark.asyncio
async def test_semantic_search_builder_receives_complete_request_contract() -> None:
    context_filter = ContextFilter(collection="knowledge")
    builder = SimpleNamespace(
        build_tool=AsyncMock(return_value=_resolved("semantic_search"))
    )
    request = _request(
        _tool("semantic_search"),
        tool_context=[IngestedArtifact(context_filter=context_filter)],
    )
    request.citation.enabled = True

    assert await SemanticSearchProcessor(builder).intercept(request)

    builder.build_tool.assert_awaited_once_with(
        model_id="contract-model",
        name="semantic_search",
        type="semantic_search_v1",
        context_filter=context_filter,
        generate_citations=True,
        validate=ToolValidationMode.EAGER,
        token_limit=98_765,
    )


@pytest.mark.asyncio
async def test_tabular_builder_receives_complete_request_contract() -> None:
    context_filter = ContextFilter(collection="tables")
    builder = SimpleNamespace(
        build_tool=AsyncMock(return_value=_resolved("tabular_analysis"))
    )
    request = _request(
        _tool("tabular_analysis"),
        tool_context=[IngestedArtifact(context_filter=context_filter)],
    )

    assert await TabularDataProcessor(builder).intercept(request)

    builder.build_tool.assert_awaited_once_with(
        model_id="contract-model",
        name="tabular_analysis",
        type="tabular_analysis_v1",
        context_filter=context_filter,
        validate=ToolValidationMode.EAGER,
        blob_visibility=BlobVisibilityMode.INTERNAL,
    )


@pytest.mark.asyncio
async def test_database_builder_receives_complete_request_contract() -> None:
    artifact = SqlDatabaseArtifact(
        connection_string="sqlite:///contract.db",
        schemas=["main"],
    )
    builder = SimpleNamespace(
        build_tool=AsyncMock(return_value=_resolved("database_query"))
    )
    request = _request(_tool("database_query"), tool_context=[artifact])

    assert await DatabaseQueryProcessor(builder).intercept(request)

    kwargs = builder.build_tool.await_args.kwargs
    assert kwargs == {
        "name": "database_query",
        "type": "database_query_v1",
        "sql_artifacts": [artifact],
        "chat_history": kwargs["chat_history"],
        "validate": ToolValidationMode.EAGER,
        "blob_visibility": BlobVisibilityMode.INTERNAL,
    }
    assert [message.role for message in kwargs["chat_history"]] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]


@pytest.mark.asyncio
async def test_web_search_builder_receives_complete_request_contract() -> None:
    builder = SimpleNamespace(
        build_tool=AsyncMock(return_value=_resolved("web_search"))
    )

    assert await WebSearchProcessor(builder).intercept(_request(_tool("web_search")))

    builder.build_tool.assert_awaited_once_with(
        model_id="contract-model",
        name="web_search",
        type="web_search_v1",
    )


@pytest.mark.asyncio
async def test_web_fetch_builder_receives_complete_request_contract() -> None:
    builder = SimpleNamespace(build_tool=Mock(return_value=_resolved("web_fetch")))

    assert await WebFetchProcessor(builder).intercept(_request(_tool("web_fetch")))

    builder.build_tool.assert_called_once_with(
        name="web_fetch",
        type="web_fetch_v1",
    )


@pytest.mark.asyncio
async def test_bash_builder_receives_complete_session_contract() -> None:
    bundle = ContentBundle(canonical_path="/mnt/skills/contract/")
    builder = SimpleNamespace(build_tool=AsyncMock(return_value=_resolved("bash")))
    request = _request(
        _tool("bash"),
        content_bundles=[bundle],
        bundles_to_remove=["/mnt/skills/old/"],
    )

    assert await BashProcessor(builder).intercept(request)

    config = builder.build_tool.await_args.args[0]
    assert config.session_id == "contract-correlation"
    assert config.extra_bundles == [bundle]
    assert config.bundles_to_remove == ["/mnt/skills/old/"]
    builder.build_tool.assert_awaited_once_with(
        config,
        name="bash",
        type="bash_v1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "builder_method"),
    [
        ("view", "build_view_tool"),
        ("str_replace", "build_str_replace_tool"),
        ("create", "build_create_tool"),
        ("insert", "build_insert_tool"),
    ],
)
async def test_text_editor_builders_receive_complete_session_contract(
    tool_name: str,
    builder_method: str,
) -> None:
    bundle = ContentBundle(canonical_path="/mnt/skills/editor/")
    builder = SimpleNamespace(
        build_view_tool=AsyncMock(return_value=_resolved("view")),
        build_str_replace_tool=AsyncMock(return_value=_resolved("str_replace")),
        build_create_tool=AsyncMock(return_value=_resolved("create")),
        build_insert_tool=AsyncMock(return_value=_resolved("insert")),
    )
    request = _request(
        _tool(tool_name),
        content_bundles=[bundle],
        bundles_to_remove=["/mnt/skills/removed/"],
    )

    assert await TextEditorProcessor(builder).intercept(request)

    method = getattr(builder, builder_method)
    config = method.await_args.args[0]
    assert config.session_id == "contract-correlation"
    assert config.extra_bundles == [bundle]
    assert config.bundles_to_remove == ["/mnt/skills/removed/"]
    method.assert_awaited_once_with(
        config,
        name=tool_name,
        type=f"{tool_name}_v1",
    )


@pytest.mark.asyncio
async def test_present_files_builder_receives_complete_request_contract() -> None:
    bundle = ContentBundle(canonical_path="/mnt/skills/present/")
    builder = SimpleNamespace(
        build_tool=AsyncMock(return_value=_resolved("present_files"))
    )
    settings = SimpleNamespace(
        code_execution=SimpleNamespace(
            tools=SimpleNamespace(present_files_enabled=True)
        )
    )

    assert await PresentFilesProcessor(builder, settings).intercept(
        _request(_tool("present_files"), content_bundles=[bundle])
    )

    builder.build_tool.assert_awaited_once_with(
        "contract-correlation",
        bundles=[bundle],
        name="present_files",
        type="present_files_v1",
    )


@pytest.mark.asyncio
async def test_present_server_builder_receives_complete_request_contract() -> None:
    builder = SimpleNamespace(
        build_tool=AsyncMock(return_value=_resolved("present_server"))
    )
    settings = SimpleNamespace(
        code_execution=SimpleNamespace(
            tools=SimpleNamespace(present_server_enabled=True)
        )
    )

    assert await PresentServerProcessor(builder, settings).intercept(
        _request(_tool("present_server"))
    )

    builder.build_tool.assert_awaited_once_with(
        "contract-correlation",
        name="present_server",
        type="present_server_v1",
    )
