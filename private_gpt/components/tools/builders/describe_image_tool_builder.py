from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton
from llama_index.core.base.llms.types import ImageBlock as LIImageBlock

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import supports_images
from private_gpt.components.multimodality.image_handler import describe_image
from private_gpt.components.tools.builders.sandbox_file_tools import (
    resolve_media_llm,
    run_over_paths,
    source_path_or_error,
    write_markdown,
)
from private_gpt.components.tools.events.adapters import DescribeImageEventAdapter
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import DESCRIBE_IMAGE_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import DESCRIBE_IMAGE_TOOL_FN
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    import asyncio

    from llama_index.core.llms import LLM

    from private_gpt.components.code_execution.base import (
        CodeExecutionSession,
        CodeExecutionSessionConfig,
    )
    from private_gpt.events.models import ResultContentBlockType

logger = logging.getLogger(__name__)


def _mime_for(source: str) -> str | None:
    """Guess an image mime type from the path's extension."""
    import filetype  # type: ignore[import-untyped]

    suffix = PurePosixPath(source).suffix.lstrip(".")
    if not suffix:
        return None
    guessed = filetype.get_type(ext=suffix)
    return str(guessed.mime) if guessed else None


async def _describe_one(
    filepath: str,
    session: CodeExecutionSession,
    image_llm: LLM,
    write_lock: asyncio.Lock,
) -> str:
    """Describe one sandbox image into the workspace, returning where it landed."""
    source = source_path_or_error(filepath, DESCRIBE_IMAGE_TOOL_NAME)
    if not await session.path_exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    raw = await session.read_file(source)
    if not raw:
        raise ValueError(f"{source} is empty.")

    block = LIImageBlock(image=raw, image_mimetype=_mime_for(source))
    description = await describe_image(image_llm, [block])
    if not description:
        raise ValueError(f"No description could be produced for {source}.")

    target = await write_markdown(session, source, description, write_lock)
    return (
        f"Described {source}. The description was left at {target} "
        f"({len(description)} characters)."
    )


@singleton
class DescribeImageToolBuilder:
    @inject
    def __init__(
        self,
        code_execution_component: CodeExecutionComponent,
        llm_component: LLMComponent,
        settings: Settings,
    ) -> None:
        self._component = code_execution_component
        self._llm_component = llm_component
        self._max_concurrency = settings.chat.preprocess.multimodal.max_concurrency

    async def build_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = DESCRIBE_IMAGE_TOOL_NAME,
        type: str = DESCRIBE_IMAGE_TOOL_NAME + "_v1",
        description: str = DESCRIBE_IMAGE_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def describe_images(
            filepaths: list[str],
        ) -> list[ResultContentBlockType]:
            session = await self._component.get_or_create_session(config)
            if session is None:
                raise ValueError("code_execution provider is not configured.")

            image_llm = resolve_media_llm(self._llm_component, supports_images, "image")

            async def _worker(filepath: str, write_lock: asyncio.Lock) -> str:
                return await _describe_one(filepath, session, image_llm, write_lock)

            return await run_over_paths(
                filepaths,
                _worker,
                self._max_concurrency,
                gerund="describing",
                past="described",
                noun="image",
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=DescribeImageEventAdapter,
            description=description,
            async_fn=describe_images,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_describe_image_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_describe_image_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(DescribeImageToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
