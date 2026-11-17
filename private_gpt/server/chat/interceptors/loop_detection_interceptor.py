import json
import logging

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from pydantic import BaseModel, Field

from private_gpt.components.chat.processors.chat_history.memory.utils.content import (
    messages_to_history_str,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.utils.loop_guard import (
    LoopGuardDirective,
    LoopGuardLevel,
)
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_LOOP_DETECTION_EXAMPLES = json.dumps(
    [
        {
            "conversation": (
                "user: Create a coaching recap skill from my notes.\n"
                "assistant: I will rewrite the skill and run the evaluation.\n"
                "tool: The evaluation still returns the same memory-oriented output.\n"
                "assistant: I will rewrite the skill and run the evaluation again.\n"
                "tool: The evaluation still returns the same memory-oriented output.\n"
                "assistant: I will rewrite the skill and run the evaluation again."
            ),
            "result": {
                "is_loop": True,
                "reason": (
                    "The assistant repeats the same rewrite-and-evaluate cycle "
                    "without changing strategy or making progress."
                ),
            },
        },
        {
            "conversation": (
                "user: Fix the failing test.\n"
                "assistant: I will run the test to inspect the failure.\n"
                "tool: TypeError: missing required argument 'name'.\n"
                "assistant: I will add the missing argument and rerun the test.\n"
                "tool: 1 passed."
            ),
            "result": {
                "is_loop": False,
                "reason": (
                    "The retry uses the error to correct the implementation and "
                    "successfully advances the task."
                ),
            },
        },
        {
            "conversation": (
                "user: Find where loop interceptors are registered and add one.\n"
                "assistant: I will inspect the interceptor service.\n"
                "tool: Found chat_interceptor_service.py.\n"
                "assistant: I will inspect the context stack API before editing.\n"
                "tool: ContextStack is immutable and supports append_layer.\n"
                "assistant: I will implement the interceptor and add tests."
            ),
            "result": {
                "is_loop": False,
                "reason": (
                    "Each assistant action gathers new information and moves the "
                    "implementation forward."
                ),
            },
        },
    ],
    indent=2,
)


class LoopDetectionResult(BaseModel):
    is_loop: bool = Field(description="Whether the assistant is entering a loop.")
    reason: str = Field(description="A concise reason for the decision.")


def _messages_since_last_user(messages: list[ChatMessage]) -> list[ChatMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == MessageRole.USER:
            return messages[index:]
    return []


@singleton
class LoopDetectionRequestInterceptor(ChatRequestLoopInterceptor):
    """Ask the LLM whether the work since the latest user message is a loop.

    Fallback for loops the deterministic repeat guard cannot see (different
    arguments, same intent). Runs once every ``interval`` assistant messages
    since the latest user message. On a detected loop, the next LLM call gets
    the history as a work log with tools removed, so the model answers with
    what it already has.
    """

    @inject
    def __init__(
        self,
        settings: Settings,
        prompt_builder_service: PromptBuilderService,
    ) -> None:
        self._interval = settings.chat.loop_detection_interval
        self._prompt_builder = prompt_builder_service

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if self._interval is None or context.phase != InterceptorPhase.BEFORE_ITERATION:
            return
        current = context.state.input.loop_guard
        if current is not None and current.level >= LoopGuardLevel.FINAL:
            return

        messages = _messages_since_last_user(context.state.input.request.to_messages())
        assistant_count = sum(
            message.role == MessageRole.ASSISTANT for message in messages
        )
        if assistant_count == 0 or assistant_count % self._interval != 0:
            return

        prompt = self._prompt_builder.create_loop_detection_prompt(
            conversation=messages_to_history_str(messages),
            examples=_LOOP_DETECTION_EXAMPLES,
        )
        try:
            result = await context.llm.astructured_predict(
                output_cls=LoopDetectionResult,
                prompt=prompt,
                llm_kwargs={"max_tokens": 512},
            )
        except Exception as error:
            logger.warning(
                "Loop detection evaluation failed; continuing normally: %s", error
            )
            return

        if not result.is_loop:
            return

        logger.info("Loop detected by classifier: %s", result.reason)
        context.state.input.loop_guard = LoopGuardDirective(
            level=LoopGuardLevel.FINAL, reason=result.reason
        )
        context.set_state(context.state)
