from __future__ import annotations

import json
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.tools.tool_names import PRESENT_SERVER_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import PRESENT_SERVER_TOOL_FN
from private_gpt.events.models import ResourceLinkBlock, TextBlock

if TYPE_CHECKING:
    from private_gpt.events.models import ResultContentBlockType


@singleton
class PresentServerToolBuilder:
    @inject
    def __init__(self, code_execution_component: CodeExecutionComponent) -> None:
        self._component = code_execution_component

    async def build_tool(
        self,
        session_id: str,
        name: str = PRESENT_SERVER_TOOL_NAME,
        type: str = PRESENT_SERVER_TOOL_NAME + "_v1",
        description: str = PRESENT_SERVER_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def present_server(
            port: int,
            service_name: str = "App",
            initial_path: str | None = None,
        ) -> list[ResultContentBlockType]:
            link = await self._component.get_session_endpoint(session_id, port)

            if link is None:
                return [
                    TextBlock(
                        text=f"No HTTP endpoint is available for port {port}. "
                        "The sandbox backend does not support HTTP ingress."
                    )
                ]

            url = link.url
            if initial_path:
                url = url.rstrip("/") + "/" + initial_path.lstrip("/")

            link_description = (
                json.dumps({"headers": link.headers}) if link.headers else None
            )

            blocks: list[ResultContentBlockType] = [
                ResourceLinkBlock(
                    uri=url,
                    name=service_name,
                    description=link_description,
                    mime_type="text/html",
                    metadata={"headers": link.headers} if link.headers else {},
                ),
                TextBlock(text=f"The service '{service_name}' is available at {url}."),
            ]

            return blocks

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description=description,
            async_fn=present_server,
            requirements=[ToolRequirements.SANDBOX],
        )
