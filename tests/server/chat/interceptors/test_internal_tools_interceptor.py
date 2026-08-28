from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedSystemConfig,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import (
    RuntimeInstructionsLayer,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
)
from private_gpt.components.engines.chat.utils.request_builder import (
    build_initial_context_stack,
    build_request_from_context_stack,
)
from private_gpt.components.tools.processors.database_query_processor import (
    DatabaseQueryProcessor,
)
from private_gpt.server.chat.interceptors.internal_tools_interceptor import (
    InternalToolRequestInterceptor,
)
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm

ORIGINAL_USER_PROMPT = "ORIGINAL_USER_PROMPT_ONLY"
RENDERED_HEADER = "RENDERED_PLATFORM_HEADER_SHOULD_NOT_LEAK"


def _make_request() -> ResolvedChatRequest:
    return ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(prompt=ORIGINAL_USER_PROMPT),
    )


@pytest.mark.asyncio
async def test_internal_tool_interceptor_keeps_rendered_system_prompt() -> None:
    """InternalToolRequestInterceptor must not rewrite the system prompt.

    It prepares tools; the database tool is responsible for deciding whether
    it wants the original user prompt or the rendered prompt.
    """
    original_request = _make_request()
    current_request = original_request.model_copy(deep=True)
    original_stack = build_initial_context_stack(original_request)
    current_stack = original_stack.append_layer(
        RuntimeInstructionsLayer(
            text=RENDERED_HEADER,
            source="platform_header",
        )
    )

    state = ChatState(
        input=ChatInputState(
            request=current_request,
            context_stack=current_stack,
        ),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
        timeline=[],
        original_input=ChatInputState(
            request=original_request, context_stack=original_stack
        ),
    )
    context = ChatInterceptorContext(
        state=state,
        llm=get_mock_function_calling_llm(["ok"]),
        phase=InterceptorPhase.VALIDATION,
        emit_fn=lambda _: None,
    )

    captured_prompts: list[object] = []
    tool = ToolSpec.from_defaults(
        name="database_query",
        type="database_query_v1",
        input_schema={"type": "object", "properties": {}},
    )

    async def fake_contextualize(req: ResolvedChatRequest) -> ResolvedChatRequest:
        captured_prompts.append(req.system.prompt)
        req.tool_config.tools = [tool]
        return req

    pipeline = MagicMock()
    pipeline.contextualize_internal_tools = AsyncMock(side_effect=fake_contextualize)

    builder = MagicMock()
    builder.seed_tool_instructions.return_value = [tool]

    interceptor = InternalToolRequestInterceptor(
        tool_pipeline=pipeline,
        prompt_builder=builder,
    )
    await interceptor.intercept(context)

    # The pipeline still sees the full rendered prompt. No global override.
    rendered_text = "\n".join(
        block.text for block in captured_prompts[0] if hasattr(block, "text")
    )
    assert ORIGINAL_USER_PROMPT in rendered_text
    assert RENDERED_HEADER in rendered_text

    # Prepared tools are persisted into the original stack so later
    # iterations/resumes do not need to re-run internal tool preparation.
    original_tools = [
        tool.name
        for tool in context.state.original_input.context_stack.all_tools()
        if tool.name
    ]
    assert "database_query" in original_tools


def test_build_request_preserves_original_prompt_separately() -> None:
    request = _make_request()
    stack = build_initial_context_stack(request).append_layer(
        RuntimeInstructionsLayer(
            text=RENDERED_HEADER,
            source="platform_header",
        )
    )

    materialized = build_request_from_context_stack(request, stack)

    original_blocks = materialized.system.get_original_prompt()
    rendered_blocks = materialized.system.get_prompt()

    original_text = "\n".join(b.text for b in original_blocks or [])
    rendered_text = "\n".join(b.text for b in rendered_blocks or [])

    assert original_text == ORIGINAL_USER_PROMPT
    assert RENDERED_HEADER not in original_text
    assert RENDERED_HEADER in rendered_text


@pytest.mark.asyncio
async def test_database_query_processor_uses_original_user_prompt() -> None:
    request = _make_request()
    request.system.original_prompt = ORIGINAL_USER_PROMPT
    request.system.prompt = [
        __import__(
            "llama_index.core.base.llms.types", fromlist=["TextBlock"]
        ).TextBlock(text=f"{ORIGINAL_USER_PROMPT}\n{RENDERED_HEADER}")
    ]
    request.tool_config.tools = [
        ToolSpec.from_defaults(
            name="database_query",
            type="database_query_v1",
            input_schema={"type": "object", "properties": {}},
        )
    ]
    # The processor needs an SQL artifact in tool context to get past the
    # guard; the builder is mocked so no real DB connection is attempted.
    from private_gpt.server.utils.artifact_input import SqlDatabaseArtifact

    request.tool_config.tools[0].context = [
        SqlDatabaseArtifact(
            connection_string="sqlite:///:memory:",
        )
    ]

    captured_chat_history: list[list[ChatMessage]] = []

    async def fake_build_tool(**kwargs: object) -> ToolSpec:
        captured_chat_history.append(kwargs["chat_history"])
        return request.tool_config.tools[0]

    builder = MagicMock()
    builder.build_tool = AsyncMock(side_effect=fake_build_tool)

    processor = DatabaseQueryProcessor(builder)
    await processor.intercept(request)

    assert captured_chat_history
    system_messages = [
        m for m in captured_chat_history[0] if m.role == MessageRole.SYSTEM
    ]
    assert len(system_messages) == 1
    system_text = "\n".join(
        b.text for b in system_messages[0].blocks if hasattr(b, "text")
    )
    assert ORIGINAL_USER_PROMPT in system_text
    assert RENDERED_HEADER not in system_text
