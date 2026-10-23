import pytest

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedContextConfig,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.tools.processors.anthropic_tool_translation_processor import (
    AnthropicToolTranslationProcessor,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_type",
    [
        "bash_20250124",
        "text_editor_20250124",
        "computer_20250124",
        "memory_20250124",
    ],
)
async def test_client_tool_translation_preserves_type_and_converges(
    tool_type: str,
) -> None:
    request = ResolvedChatRequest(
        messages=[],
        tool_config=ResolvedToolConfig(
            tools=[
                ToolSpec(
                    name=tool_type.rsplit("_", 1)[0],
                    type=tool_type,
                    input_schema=None,
                )
            ]
        ),
        context=ResolvedContextConfig(correlation_id="correlation-id"),
    )
    processor = AnthropicToolTranslationProcessor()

    assert await processor.intercept(request) is True
    assert request.tool_config.tools[0].type == tool_type
    assert request.tool_config.tools[0].description
    assert request.tool_config.tools[0].input_schema

    assert await processor.intercept(request) is False
