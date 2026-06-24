from __future__ import annotations

from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.tools.tool_names import BASH_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import BASH_TOOL_FN
from private_gpt.events.models import TextBlock
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.sandbox.content_bundle import ContentBundle
    from private_gpt.events.models import ResultContentBlockType


def _truncate_output(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + "\n...[truncated]"


@singleton
class BashToolBuilder:
    @inject
    def __init__(
        self,
        code_execution_component: CodeExecutionComponent,
        settings: Settings,
    ) -> None:
        self._component = code_execution_component
        self._settings = settings

    async def build_tool(
        self,
        session_id: str,
        bundles: list[ContentBundle] | None = None,
        name: str = BASH_TOOL_NAME,
        type: str = BASH_TOOL_NAME + "_v1",
        description: str = BASH_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        session = await self._component.get_or_create_session(
            session_id, extra_bundles=bundles or None
        )
        if session is None:
            raise ValueError("code_execution provider is not configured.")

        async def run_bash(
            command: str,
            timeout: int | None = None,
            restart: bool = False,
        ) -> list[ResultContentBlockType]:
            result = await session.execute_bash(
                command,
                timeout=timeout,
                restart=restart,
            )
            sections = [f"exit_code: {result.exit_code}"]
            if result.stdout:
                sections.append(f"stdout:\n{result.stdout}")
            if result.stderr:
                sections.append(f"stderr:\n{result.stderr}")
            output = _truncate_output(
                "\n\n".join(sections),
                self._settings.code_execution.max_output_bytes,
            )
            return [TextBlock(text=output)]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description=description,
            async_fn=run_bash,
            requirements=[ToolRequirements.SANDBOX],
        )
