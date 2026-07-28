from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.tools.events.adapters import BashCodeExecutionEventAdapter
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import BASH_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import BASH_TOOL_FN
from private_gpt.components.tools.utils import truncate_output
from private_gpt.di import get_global_injector
from private_gpt.events.models import BashCodeExecutionResultBlock
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
    from private_gpt.events.models import ResultContentBlockType


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
        config: CodeExecutionSessionConfig,
        name: str = BASH_TOOL_NAME,
        type: str = BASH_TOOL_NAME + "_v1",
        description: str = BASH_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def run_bash(
            command: str,
            timeout: int | None = None,
            restart: bool = False,
        ) -> list[ResultContentBlockType]:
            session = await self._component.get_or_create_session(config)
            if session is None:
                raise ValueError("code_execution provider is not configured.")

            result = await session.execute_bash(
                command,
                timeout=timeout,
                restart=restart,
            )
            return [
                BashCodeExecutionResultBlock(
                    stdout=truncate_output(
                        result.stdout,
                        self._settings.code_execution.max_output_bytes,
                    ),
                    stderr=truncate_output(
                        result.stderr,
                        self._settings.code_execution.max_output_bytes,
                    ),
                    return_code=result.exit_code,
                )
            ]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=BashCodeExecutionEventAdapter,
            description=description,
            async_fn=run_bash,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_bash_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_bash_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(BashToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
