import asyncio
import base64
import logging
import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TransformComponent

from private_gpt.components.concurrency.registry import create_semaphore_manager
from private_gpt.components.concurrency.semaphore_manager import SemaphoreManager
from private_gpt.components.llm.llm_helper import supports_images
from private_gpt.components.multimodality.image_handler import describe_image
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import TransformationReadersSettings

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_DELAY = 2.0
_JITTER = (5.0, 60.0)
_MAX_TEXT_CONTEXT_CHARS = 10_000
_DEFAULT_TIMEOUT = 365 * 24 * 60 * 60


class ExtractDocumentContentFromImage(TransformComponent):
    """Extract complete content using full page imagen docs."""

    def __init__(
        self, reader_settings: TransformationReadersSettings, **kwargs: Any
    ) -> None:
        super().__init__()
        self._reader_settings = reader_settings
        self._kwargs = kwargs
        self._image_regex = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)")

    @classmethod
    def from_defaults(
        cls,
        reader_settings: TransformationReadersSettings,
        retry_delay: float = _DEFAULT_RETRY_DELAY,
        retry_jitter: tuple[float, float] = _JITTER,
    ) -> "ExtractDocumentContentFromImage":
        kwargs: Any = {
            "num_max_retries": reader_settings.vision.retry_number,
            "retry_delay": retry_delay,
            "retry_jitter": retry_jitter,
        }
        return cls(reader_settings=reader_settings, **kwargs)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return asyncio.run(self.acall(nodes=nodes, **kwargs))

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        from private_gpt.components.llm.llm_component import LLMComponent

        injector = get_global_injector()
        llm_component = injector.get(LLMComponent)
        prompt_builder = injector.get(PromptBuilderService)

        vlm_instance = next(
            llm_component.filter(lambda llm, cfg: supports_images(llm, cfg)),
            None,
        )

        if not vlm_instance:
            return nodes

        vlm: LLM = vlm_instance[0]

        # Prepare kwargs for image description
        final_kwargs: Any = {
            **self._kwargs,
            **kwargs,
            "callback_manager": vlm.callback_manager,
            "timeout": _DEFAULT_TIMEOUT,
        }

        if not self._reader_settings.vision.enable_semaphore:
            return await self._execute(
                nodes=nodes, vlm=vlm, prompt_builder=prompt_builder, **final_kwargs
            )

        async with create_semaphore_manager(
            max_concurrency=self._reader_settings.vision.max_concurrent,
            queue_key="doc_extraction",
        ) as semaphore_manager:
            return await self._execute(
                nodes=nodes,
                vlm=vlm,
                prompt_builder=prompt_builder,
                semaphore_manager=semaphore_manager,
                **final_kwargs,
            )

    async def _execute(
        self,
        nodes: Sequence[BaseNode],
        vlm: LLM,
        prompt_builder: PromptBuilderService,
        semaphore_manager: SemaphoreManager | None = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        tasks = [
            semaphore_manager.execute(
                task_func=self._extract_doc_content,
                priority=idx,
                node=node,
                vlm=vlm,
                prompt_builder=prompt_builder,
                **kwargs,
            )
            if semaphore_manager
            else self._extract_doc_content(
                node=node,
                vlm=vlm,
                prompt_builder=prompt_builder,
                **kwargs,
            )
            for idx, node in enumerate(nodes)
        ]
        processed_nodes = await asyncio.gather(*tasks)
        return list(processed_nodes)

    async def _extract_doc_content(
        self,
        node: BaseNode,
        vlm: LLM,
        prompt_builder: PromptBuilderService,
        **kwargs: Any,
    ) -> BaseNode:
        doc_image_b64 = node.metadata.get("slide_image")
        if not doc_image_b64:
            return node

        try:
            image_block = ImageBlock(
                image=base64.b64decode(doc_image_b64),
                image_mimetype="image/jpeg",
            )
        except Exception as e:
            logger.warning("Failed to decode doc image metadata: %s", e)
            self._clean_doc_metadata(node)
            return node

        prompt = prompt_builder.create_document_image_extract_prompt()
        user_query = prompt.format().strip()

        try:
            extracted_content = await describe_image(
                image_multimodal_llm=vlm,
                image_blocks=[image_block],
                user_query=user_query,
                enable_preprocessing=False,
                enable_evaluation=self._reader_settings.vision.enable_evaluation,
                max_iterations=self._reader_settings.vision.max_iterations,
                extraction_type_override="mixed",
                **kwargs,
            )
            if extracted_content and extracted_content.strip():
                node.set_content(extracted_content.strip() + "\n")
        except Exception as e:
            logger.exception("Failed to extract doc content from image+text: %s", e)
        finally:
            self._clean_doc_metadata(node)

        return node

    @staticmethod
    def _clean_doc_metadata(node: BaseNode) -> None:
        if "slide_image" in node.metadata:
            del node.metadata["slide_image"]
