from __future__ import annotations

import asyncio
import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.ingestion.ingestion_scheduler import (
    IngestionSchedulerFactory,
)
from private_gpt.components.tools.builders.sandbox_file_tools import (
    run_over_paths,
    source_path_or_error,
    write_markdown,
)
from private_gpt.components.tools.events.adapters import ConvertDocumentsEventAdapter
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import CONVERT_DOCUMENTS_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import CONVERT_DOCUMENTS_TOOL_FN
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import (
        CodeExecutionSession,
        CodeExecutionSessionConfig,
    )
    from private_gpt.events.models import ResultContentBlockType
    from private_gpt.events.models._content_blocks import DocumentConverter

logger = logging.getLogger(__name__)


async def _convert_one(
    filepath: str,
    session: CodeExecutionSession,
    convert_service: DocumentConverter,
    write_lock: asyncio.Lock,
) -> str:
    """Convert one sandbox file to markdown in the workspace, return the target.

    The expensive half runs concurrently: ``read_file`` is awaitable and
    ``bytes_to_text`` is offloaded to a thread, so a batch of documents converts
    in parallel rather than one after another.
    """
    source = source_path_or_error(filepath, CONVERT_DOCUMENTS_TOOL_NAME)
    if not await session.path_exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    raw = await session.read_file(source)
    extension = PurePosixPath(source).suffix or ".txt"
    text = await asyncio.to_thread(convert_service.bytes_to_text, raw, extension, False)
    if not text:
        raise ValueError(f"No content could be extracted from {source}.")

    target = await write_markdown(session, source, text, write_lock)
    return (
        f"Converted {source} to markdown. The content was left at {target} "
        f"({len(text)} characters)."
    )


@singleton
class ConvertDocumentsToolBuilder:
    @inject
    def __init__(
        self,
        code_execution_component: CodeExecutionComponent,
        scheduler_factory: IngestionSchedulerFactory,
        settings: Settings,
    ) -> None:
        self._component = code_execution_component
        self._scheduler_factory = scheduler_factory
        self._max_concurrency = settings.chat.preprocess.documents.max_concurrency

    async def build_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = CONVERT_DOCUMENTS_TOOL_NAME,
        type: str = CONVERT_DOCUMENTS_TOOL_NAME + "_v1",
        description: str = CONVERT_DOCUMENTS_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def convert_documents(
            filepaths: list[str],
        ) -> list[ResultContentBlockType]:
            session = await self._component.get_or_create_session(config)
            if session is None:
                raise ValueError("code_execution provider is not configured.")

            convert_service = self._scheduler_factory.get()

            async def _worker(filepath: str, write_lock: asyncio.Lock) -> str:
                return await _convert_one(
                    filepath, session, convert_service, write_lock
                )

            return await run_over_paths(
                filepaths,
                _worker,
                self._max_concurrency,
                gerund="converting",
                past="converted",
                noun="document",
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=ConvertDocumentsEventAdapter,
            description=description,
            async_fn=convert_documents,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_convert_documents_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_convert_documents_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(ConvertDocumentsToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
