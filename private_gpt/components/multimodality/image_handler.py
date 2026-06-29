import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from llama_index.core.base.llms.types import ImageBlock, TextBlock
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field
from workflows.resource import ResourceManager

from private_gpt.artifact_index.artifact_exception import ModelNotAvailableError
from private_gpt.components.llm.models import MODEL_NOT_AVAILABLE_EXCEPTION_TYPES
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.readers.nodes.image_node import IMAGE_NOT_PROCESSABLE
from private_gpt.components.workflows.types import AnyContext
from private_gpt.di import get_global_injector
from private_gpt.events.event_errors import Errors
from private_gpt.utils.dependencies import format_missing_dependency_message
from private_gpt.utils.retry import retry_context

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import ContentBlock
    from llama_index.core.program.utils import FlexibleModel
    from workflows.handler import WorkflowHandler

    from private_gpt.components.concurrency.semaphore_manager import SemaphoreManager

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_NUMBER = 3
_JITTER = (15.0, 30.0)


SKIPPABLE_TYPES = ["icon", "blank", "corrupted"]


class ExtractionStrategy(BaseModel):
    type: Literal[
        "picture",
        "text",
        "icon",
        "table",
        "form",
        "diagram",
        "chart",
        "mixed",
        "blank",
        "corrupted",
    ] = Field(description="Primary content type detected in the image")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the detection (0-1)"
    )
    language: str = Field(default="en", description="Detected language code")
    has_structure: bool = Field(
        default=False, description="Whether image contains structured elements"
    )
    increase_contrast: bool | None = Field(
        default=None, description="Whether to enhance contrast"
    )


class ExtractionContent(BaseModel):
    markdown: str = Field(description="Extracted content in Markdown format")
    is_complete: bool | None = Field(description="True if extraction is finished")


class ExtractionEvaluation(BaseModel):
    score: float = Field(description="Overall quality score")
    issues_found: list[str] = Field(
        default_factory=list, description="Issues identified"
    )


class ImageProcessingInputEvent(StartEvent):
    image_blocks: list[ImageBlock]
    user_query: str | None = None
    max_iterations: int = 3
    enable_preprocessing: bool = True
    enable_evaluation: bool = True
    skip_strategy_inference: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkflowInitializedEvent(Event):
    image_blocks: list[ImageBlock]


class StrategyInferredEvent(Event):
    strategy: ExtractionStrategy | None


class ImagesPreprocessedEvent(Event):
    processed_blocks: list[ImageBlock]


class ContentExtractedEvent(Event):
    content: str
    is_complete: bool
    iteration: int
    all_results: list[str]


class ContentEvaluatedEvent(Event):
    final_content: str
    evaluation: ExtractionEvaluation | None = None


class FinalizeContentEvent(Event):
    final_content: str
    evaluation: ExtractionEvaluation | None = None


class ImageProcessingResultEvent(StopEvent):
    description: str
    evaluation: ExtractionEvaluation | None = None


class ImageProcessingWorkflow(Workflow):
    def __init__(
        self,
        image_multimodal_llm: LLM,
        prompt_builder: PromptBuilderService | None = None,
        callback_manager: CallbackManager | None = None,
        timeout: float | None = 360000.0,
        disable_validation: bool = False,
        verbose: bool = False,
        resource_manager: ResourceManager | None = None,
        num_concurrent_runs: int | None = None,
        num_max_retries: int = _DEFAULT_RETRY_NUMBER,
        retry_jitter: tuple[float, float] = _JITTER,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            timeout=timeout,
            disable_validation=disable_validation,
            verbose=verbose,
            resource_manager=resource_manager,
            num_concurrent_runs=num_concurrent_runs,
        )
        self._llm = image_multimodal_llm
        self._prompt_builder = prompt_builder or get_global_injector().get(
            PromptBuilderService
        )
        self._kwargs = kwargs

        if callback_manager:
            self._llm.callback_manager = callback_manager

        self._num_max_retries = num_max_retries
        self._retry_jitter = retry_jitter

    @step
    async def init_workflow(
        self, ctx: AnyContext, ev: ImageProcessingInputEvent
    ) -> WorkflowInitializedEvent:
        await ctx.store.set("image_blocks", ev.image_blocks)
        await ctx.store.set("user_query", ev.user_query)
        await ctx.store.set("max_iterations", ev.max_iterations)
        await ctx.store.set("enable_preprocessing", ev.enable_preprocessing)
        await ctx.store.set("enable_evaluation", ev.enable_evaluation)
        await ctx.store.set("skip_strategy_inference", ev.skip_strategy_inference)
        await ctx.store.set("kwargs", ev.kwargs)
        await ctx.store.set("results", [])
        await ctx.store.set("iteration", 1)

        return WorkflowInitializedEvent(image_blocks=ev.image_blocks)

    @step
    async def infer_strategy(
        self, ctx: AnyContext, ev: WorkflowInitializedEvent
    ) -> StrategyInferredEvent | ImagesPreprocessedEvent:

        enable_preprocessing = await ctx.store.get("enable_preprocessing")
        skip_strategy_inference = await ctx.store.get("skip_strategy_inference")
        kwargs = await ctx.store.get("kwargs")

        if skip_strategy_inference:
            strategy = None
        else:
            iteration = await ctx.store.get("iteration", 1)
            strategy = await self._infer_strategy(
                ev.image_blocks, seed=iteration, **kwargs
            )

        await ctx.store.set("strategy", strategy)

        return (
            StrategyInferredEvent(strategy=strategy)
            if enable_preprocessing
            else ImagesPreprocessedEvent(processed_blocks=ev.image_blocks)
        )

    @step
    async def preprocess_images(
        self, ctx: AnyContext, ev: StrategyInferredEvent
    ) -> ImagesPreprocessedEvent:
        image_blocks = await ctx.store.get("image_blocks")
        enable_preprocessing = await ctx.store.get("enable_preprocessing")

        processed_blocks = image_blocks
        if enable_preprocessing and ev.strategy and ev.strategy.increase_contrast:
            processed_blocks = [
                self._preprocess_image(block, ev.strategy) for block in image_blocks
            ]

        await ctx.store.set("strategy", ev.strategy)
        return ImagesPreprocessedEvent(processed_blocks=processed_blocks)

    @step
    async def extract_content(
        self, ctx: AnyContext, ev: ImagesPreprocessedEvent
    ) -> ContentEvaluatedEvent | ImagesPreprocessedEvent:
        user_query = await ctx.store.get("user_query")
        kwargs = await ctx.store.get("kwargs")
        strategy = await ctx.store.get("strategy")
        max_iterations = await ctx.store.get("max_iterations")

        results = await ctx.store.get("results", [])
        iteration = await ctx.store.get("iteration", 1)
        issues_found = await ctx.store.get("issues_found", [])

        skippable = (
            strategy is not None
            and strategy.type in SKIPPABLE_TYPES
            and strategy.confidence > 0.6
        )
        if skippable:
            await ctx.store.set("iteration", max_iterations + 1)
            return ContentEvaluatedEvent(final_content=IMAGE_NOT_PROCESSABLE)

        if iteration > max_iterations:
            final_content = " ".join(results) if results else ""
            return ContentEvaluatedEvent(final_content=final_content)

        continue_content = " ".join(results) if results else None
        template = self._prompt_builder.create_image_interpretation_prompt(
            user_query=user_query,
            last_content=continue_content,
            extraction_type=strategy.type if strategy else None,
            confidence=strategy.confidence if strategy else None,
            language=strategy.language if strategy else None,
            has_structure=strategy.has_structure if strategy else None,
            errors=issues_found or None,
        )

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, blocks=[TextBlock(text=template.format())]
            ),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text="Extract content from this image:"),
                    *ev.processed_blocks,
                ],
            ),
        ]

        response = await self._astructured_chat(
            ExtractionContent,
            messages,
            seed=iteration,
            allow_flexible=True,
            **kwargs,
        )

        content = response.markdown if hasattr(response, "markdown") else ""
        replace_content = bool(issues_found)
        iteration = iteration + 1
        is_complete = (
            getattr(response, "is_complete", None) or iteration > max_iterations
        )

        results = [] if replace_content else results
        results.append(content)

        await ctx.store.set("results", results)
        await ctx.store.set("iteration", iteration)
        await ctx.store.set("issues_found", [])

        if is_complete:
            final_content = " ".join(results) if results else ""
            return ContentEvaluatedEvent(final_content=final_content)

        return ImagesPreprocessedEvent(
            processed_blocks=await ctx.store.get("image_blocks")
        )

    @step
    async def evaluate(
        self, ctx: AnyContext, ev: ContentEvaluatedEvent
    ) -> ImagesPreprocessedEvent | FinalizeContentEvent:
        image_blocks = await ctx.store.get("image_blocks")
        enable_evaluation: bool = await ctx.store.get("enable_evaluation")
        max_iterations: int = await ctx.store.get("max_iterations")
        iteration: int = await ctx.store.get("iteration", 1)
        kwargs = await ctx.store.get("kwargs")

        if not enable_evaluation or iteration > max_iterations:
            return FinalizeContentEvent(
                final_content=ev.final_content, evaluation=ev.evaluation
            )

        final_content = ev.final_content
        evaluation = await self._evaluate_content(
            final_content, image_blocks, seed=iteration, **kwargs
        )

        if evaluation and (evaluation.score < 0.7 or evaluation.issues_found):
            await ctx.store.set("issues_found", evaluation.issues_found)

            return ImagesPreprocessedEvent(
                processed_blocks=await ctx.store.get("image_blocks")
            )

        return FinalizeContentEvent(final_content=final_content, evaluation=evaluation)

    @step
    async def finalize(
        self, ctx: AnyContext, ev: FinalizeContentEvent
    ) -> ImageProcessingResultEvent:

        user_query = await ctx.store.get("user_query")
        final_content = ev.final_content
        evaluation = ev.evaluation

        response = (
            "\n\n"
            + self._prompt_builder.create_image_interpretation_response(
                user_query=user_query, content=final_content
            )
            .format()
            .strip()
            + "\n\n"
        )
        if not response.strip():
            response = IMAGE_NOT_PROCESSABLE

        return ImageProcessingResultEvent(description=response, evaluation=evaluation)

    async def _astructured_chat(
        self,
        response_model: type[BaseModel],
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Any:
        try:
            async with retry_context(
                tries=self._num_max_retries,
                jitter=self._retry_jitter,
                logger=logger,
            ) as retry:
                seed = kwargs.pop("seed", None) or 0
                semaphore_manager: SemaphoreManager | None = kwargs.pop(
                    "semaphore_manager", None
                )
                count = 0
                max_iterations = kwargs.pop("max_iterations", 3)

                async def _call() -> Any:
                    nonlocal count
                    count += 1

                    if not hasattr(self._llm, "astructured_chat"):
                        raise NotImplementedError(
                            "LLM does not support structured chat."
                        )

                    new_kwargs = kwargs.copy()
                    new_kwargs["seed"] = str(seed) + str(count)

                    try:
                        current_messages = messages
                        if count > 1:
                            current_messages = self._reduce_images_in_messages(
                                messages, count - 1
                            )
                            logger.info(
                                f"Retry {count}: Reduced image quality (iteration {count - 1}/{max_iterations})"
                            )

                        return await self._llm.astructured_chat(
                            response_model, current_messages, **new_kwargs
                        )
                    except Errors.RequestTooLarge as e:
                        logger.warning(
                            f"Request too large on attempt {count}, will retry with reduced quality"
                        )
                        if count >= self._num_max_retries:
                            return e
                        raise

                async def _call_with_semaphore() -> Any:
                    if semaphore_manager:
                        return await semaphore_manager.execute(
                            task_func=_call, priority=0
                        )
                    return await _call()

                result = await retry(_call_with_semaphore)  # type: ignore[call-arg]
                if isinstance(result, Exception):
                    raise result
                return result
        except MODEL_NOT_AVAILABLE_EXCEPTION_TYPES as e:
            raise ModelNotAvailableError(
                "Model server is not available or request failed."
            ) from e

    async def _infer_strategy(
        self, image_blocks: list[ImageBlock], **kwargs: Any
    ) -> ExtractionStrategy:
        strategy_prompt = self._prompt_builder.create_image_strategy_prompt()

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                blocks=[TextBlock(text=strategy_prompt.format())],
            ),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text="Analyze the following image:"),
                    *image_blocks,
                ],
            ),
        ]

        strategy: ExtractionStrategy | FlexibleModel = await self._astructured_chat(
            ExtractionStrategy, messages, **kwargs
        )
        if not isinstance(strategy, ExtractionStrategy):
            strategy = ExtractionStrategy(**strategy.model_dump())

        return strategy

    def _preprocess_image(
        self, image_block: ImageBlock, strategy: ExtractionStrategy
    ) -> ImageBlock:
        if not strategy.increase_contrast:
            return image_block

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            image_b = image_block.resolve_image()
            image_b.seek(0)
            image_bytes = image_b.read()

            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return image_block

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced)

            if strategy.language.lower() in ["jp", "cn", "zh", "kr"]:
                thresh = cv2.adaptiveThreshold(
                    denoised,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )
            else:
                _, thresh = cv2.threshold(
                    denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            _, buffer = cv2.imencode(".png", thresh)
            return ImageBlock(image=buffer.tobytes(), image_mimetype="image/png")

        except ImportError:
            logger.warning(
                format_missing_dependency_message(
                    "Image preprocessing",
                    extras="media",
                )
            )
            return image_block
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image_block

    def _reduce_images_in_messages(
        self,
        messages: list[ChatMessage],
        iteration: int,
    ) -> list[ChatMessage]:
        reduced_messages = []
        for message in messages:
            reduced_blocks: list[ContentBlock] = []
            for block in message.blocks:
                if isinstance(block, ImageBlock):
                    reduced_block = self._reduce_image_quality(block, iteration)
                    reduced_blocks.append(reduced_block)
                else:
                    reduced_blocks.append(block)

            reduced_message = ChatMessage(
                role=message.role,
                blocks=reduced_blocks,
                additional_kwargs=message.additional_kwargs,
            )
            reduced_messages.append(reduced_message)

        return reduced_messages

    def _reduce_image_quality(
        self,
        image_block: ImageBlock,
        iteration: int,
    ) -> ImageBlock:
        try:
            import base64
            from io import BytesIO

            from PIL import Image

            if TYPE_CHECKING:
                from PIL.Image import Image as PILImage

            bytesio = image_block.resolve_image(as_base64=True)
            image_data = base64.b64decode(bytesio.read().decode("utf-8"))

            image: PILImage = Image.open(BytesIO(image_data))
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            reduction_per_iteration = 0.15
            scale_factor = 1.0 - (iteration * reduction_per_iteration)
            scale_factor = max(0.6, scale_factor)  # Never go below 60% of original

            # Quality reduction: start at 100, reduce by 15 points per iteration
            quality = max(60, int(100 - (iteration * 15)))

            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)

            if new_width < image.width or new_height < image.height:
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            output.seek(0)

            return ImageBlock(image=output.getvalue(), image_mimetype="image/jpeg")

        except ImportError:
            logger.warning(
                "%s Skipping image quality reduction.",
                format_missing_dependency_message(
                    "Image quality reduction",
                    extras="media",
                ),
            )
            return image_block
        except Exception as e:
            logger.error(f"Image quality reduction failed: {e}")
            return image_block

    async def _evaluate_content(
        self, content: str, image_blocks: ImageBlock, **kwargs: Any
    ) -> ExtractionEvaluation | None:
        if not content:
            return None

        eval_template = self._prompt_builder.create_image_evaluation_prompt()
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, blocks=[TextBlock(text=eval_template.format())]
            ),
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(
                        text=content,
                    ),
                    *image_blocks,
                ],
            ),
        ]

        if not hasattr(self._llm, "astructured_chat"):
            raise NotImplementedError("LLM does not support structured chat.")

        result: ExtractionEvaluation | FlexibleModel = await self._astructured_chat(
            ExtractionEvaluation, messages, **kwargs
        )
        if not isinstance(result, ExtractionEvaluation):
            result = ExtractionEvaluation(**result.model_dump())

        return result


async def describe_image(
    image_multimodal_llm: LLM,
    image_blocks: list[ImageBlock] | None = None,
    user_query: str | None = None,
    max_iterations: int = 2,
    enable_preprocessing: bool = True,
    enable_evaluation: bool = True,
    skip_strategy_inference: bool = False,
    **kwargs: Any,
) -> str | None:
    if not image_blocks:
        return None

    workflow = ImageProcessingWorkflow(image_multimodal_llm, **kwargs)
    handler: WorkflowHandler | None = None

    try:
        result = cast(
            ImageProcessingResultEvent,
            await workflow.run(
                image_blocks=image_blocks,
                user_query=user_query,
                max_iterations=max_iterations,
                enable_preprocessing=enable_preprocessing,
                enable_evaluation=enable_evaluation,
                kwargs=kwargs,
            ),
        )

        return result.description if result else None
    except asyncio.CancelledError as e:
        if handler:
            await handler.cancel_run()
        raise e


async def process_images_in_message(
    image_multimodal_llm: LLM,
    message: ChatMessage,
    user_query: str | None = None,
    enable_preprocessing: bool = True,
    enable_evaluation: bool = True,
    **kwargs: Any,
) -> str | None:
    image_blocks = [block for block in message.blocks if isinstance(block, ImageBlock)]
    if not image_blocks:
        return message.content or ""

    return await describe_image(
        image_multimodal_llm,
        image_blocks,
        user_query,
        enable_preprocessing=enable_preprocessing,
        enable_evaluation=enable_evaluation,
        **kwargs,
    )
