import asyncio
import base64
import logging
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from private_gpt.components.concurrency.registry import create_semaphore_manager
from private_gpt.components.concurrency.semaphore_manager import SemaphoreManager
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import supports_images
from private_gpt.components.multimodality.image_handler import describe_image
from private_gpt.components.readers.nodes.image_node import IMAGE_PLACEHOLDER
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import TransformationReadersSettings

logger = logging.getLogger(__name__)


_JITTER = (5.0, 60.0)
_DEFAULT_RETRY_DELAY = 2.0
_DEFAULT_TIMEOUT = 365 * 24 * 60 * 60


class DescribeImageTransform(TransformComponent):
    def __init__(
        self,
        reader_settings: TransformationReadersSettings,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._regex = re.compile(r"!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)")
        self._description_cache: dict[str, str] = {}
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._reader_settings = reader_settings
        self._kwargs = kwargs

    @classmethod
    def from_defaults(
        cls,
        reader_settings: TransformationReadersSettings,
        retry_delay: float = _DEFAULT_RETRY_DELAY,
        retry_jitter: tuple[float, float] = _JITTER,
    ) -> "DescribeImageTransform":
        kwargs: Any = {
            "num_max_retries": reader_settings.vision.retry_number,
            "retry_delay": retry_delay,
            "retry_jitter": retry_jitter,
        }
        return cls(
            reader_settings=reader_settings,
            max_concurrency=reader_settings.vision.max_concurrent,
            batch_size=reader_settings.vision.batch_size,
            **kwargs,
        )

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return asyncio.run(self.acall(nodes, **kwargs))

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        # Since we are executing async code since sync entrypoint,
        # we need to get the LLMComponent instance from the global injector
        llm_component = get_global_injector().get(LLMComponent)
        vlm_instance = next(
            llm_component.filter(lambda llm, cfg: supports_images(llm, cfg)),
            None,
        )

        if not vlm_instance:
            return nodes

        vlm = vlm_instance[0]

        # Prepare kwargs for image description
        final_kwargs: Any = {
            **self._kwargs,
            **kwargs,
            "callback_manager": vlm.callback_manager,
            "timeout": _DEFAULT_TIMEOUT,
        }

        if not self._reader_settings.vision.enable_semaphore:
            return await self._execute(nodes=nodes, vlm=vlm, **final_kwargs)

        async with create_semaphore_manager(
            max_concurrency=self._max_concurrency,
            queue_key="image_descriptor",
        ) as semaphore_manager:
            return await self._execute(
                nodes=nodes,
                vlm=vlm,
                semaphore_manager=semaphore_manager,
                **final_kwargs,
            )

    async def _execute(
        self,
        nodes: Sequence[BaseNode],
        vlm: LLM,
        semaphore_manager: SemaphoreManager | None = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        unique_images = self._extract_unique_images(nodes)
        await self._process_images_in_batches(
            vlm=vlm,
            unique_images=unique_images,
            semaphore_manager=semaphore_manager,
            **kwargs,
        )
        enhanced_nodes = [self._apply_cached_descriptions(node) for node in nodes]
        return enhanced_nodes

    def _extract_unique_images(
        self, nodes: Sequence[BaseNode]
    ) -> dict[str, dict[str, Any]]:
        unique_images = {}

        for node in nodes:
            content = node.get_content(MetadataMode.NONE)
            matches = list(re.finditer(self._regex, content))

            for match in matches:
                alt_text, mime_type, b64_content = match.groups()

                if b64_content in unique_images or not self._is_valid_base64(
                    b64_content
                ):
                    continue

                unique_images[b64_content] = {
                    "alt_text": alt_text.replace("Image", ""),
                    "mime_type": mime_type,
                    "processed": b64_content in self._description_cache,
                }

        return unique_images

    async def _process_images_in_batches(
        self,
        unique_images: dict[str, dict[str, Any]],
        vlm: LLM,
        semaphore_manager: SemaphoreManager | None = None,
        **kwargs: Any,
    ) -> None:
        uncached_images = {
            b64: info for b64, info in unique_images.items() if not info["processed"]
        }
        if not uncached_images:
            return

        b64_contents = list(uncached_images.keys())
        batch_size = self._batch_size or len(b64_contents)
        batches = range(0, len(b64_contents), batch_size)
        total_batches = len(batches)
        for batch_num, i in enumerate(batches, start=1):
            logger.info("Describing images: batch %d/%d", batch_num, total_batches)
            batch = b64_contents[i : i + batch_size]
            batch_tasks = [
                self._describe_single_image(
                    vlm=vlm,
                    b64_content=b64_content,
                    image_info=unique_images[b64_content],
                    semaphore_manager=semaphore_manager,
                    **kwargs,
                )
                for b64_content in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for b64_content, result in zip(batch, batch_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to process image: {result}")
                    self._description_cache[b64_content] = (
                        unique_images[b64_content]["alt_text"] or IMAGE_PLACEHOLDER
                    )
                else:
                    self._description_cache[b64_content] = str(result)

    async def _describe_single_image(
        self,
        vlm: LLM,
        b64_content: str,
        image_info: dict[str, Any],
        semaphore_manager: SemaphoreManager | None = None,
        **kwargs: Any,
    ) -> str:
        image_block = ImageBlock(
            image=base64.b64decode(b64_content),
            image_mimetype=image_info["mime_type"],
            detail=image_info["alt_text"] or None,
        )

        result: str | None = None

        if vlm:
            extra_kwargs: dict[str, Any] = {}
            if self._reader_settings.vision.enable_semaphore:
                extra_kwargs["semaphore_manager"] = semaphore_manager

            result = await describe_image(
                image_multimodal_llm=vlm,
                image_blocks=[image_block],
                enable_evaluation=self._reader_settings.vision.enable_evaluation,
                max_iterations=self._reader_settings.vision.max_iterations,
                **extra_kwargs,
                **kwargs,
            )

        return result or image_info["alt_text"] or IMAGE_PLACEHOLDER

    def _apply_cached_descriptions(self, node: BaseNode) -> BaseNode:
        content = node.get_content(MetadataMode.NONE)
        enhanced_content = self._replace_images_with_descriptions(content)
        node.set_content(enhanced_content)

        return node

    def _replace_images_with_descriptions(self, content: str) -> str:
        def replace_match(match: re.Match[str]) -> str:
            b64_content = match.groups()[2]
            return self._description_cache.get(b64_content, match.group(0))

        return re.sub(self._regex, replace_match, content)

    @staticmethod
    def _is_valid_base64(s: str) -> bool:
        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
