import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Literal

from llama_index.core import BasePromptTemplate, PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.core.llms import LLM
from pydantic import BaseModel, TypeAdapter

from private_gpt.components.chat.processors.chat_history.memory.strategies.base_strategy import (
    BaseMemoryStrategy,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.content import (
    messages_to_history_str,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.format import (
    guarantee_valid_message_sequence,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.repairs import (
    repair_with_tools,
    repair_without_tools,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.splitting import (
    get_assistant_tool_pair_mesages,
    get_user_blocks,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.tools.builders.summary_builder import (
    SummarizeWorkflowBuilder,
)
from private_gpt.di import get_global_injector
from private_gpt.utils.batches import aiter_batch
from private_gpt.utils.tokens import (
    MessageInputProtocol,
    async_tokenizer,
    estimate_token_count,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine


MAX_CONDENSE_ITERATIONS = 2


class _SimplifiedTextMessage(BaseModel):
    """A simplified text message model for summarization for the LLM guided decoding."""

    role: str
    content: str


class CondenserContextMemoryStrategy(BaseMemoryStrategy):
    def __init__(
        self,
        summarize_workflow_builder: SummarizeWorkflowBuilder | None = None,
        prompt_builder_service: PromptBuilderService | None = None,
        llm_component: LLMComponent | None = None,
        message_to_input: MessageInputProtocol | None = None,
        **kwargs: dict[str, Any],
    ):
        self.message_to_input = message_to_input
        self.builder = summarize_workflow_builder or get_global_injector().get(
            SummarizeWorkflowBuilder
        )
        self.llm_component = llm_component or get_global_injector().get(LLMComponent)
        self.prompt_builder_service = (
            prompt_builder_service or get_global_injector().get(PromptBuilderService)
        )

    def _split_conversation(
        self,
        chat_history: list[ChatMessage],
    ) -> tuple[list[ChatMessage], ChatMessage, list[ChatMessage]]:
        """Split conversation into left and right parts."""
        for i in range(len(chat_history) - 1, -1, -1):
            if chat_history[i].role == "user":
                return chat_history[:i], chat_history[i], chat_history[i + 1 :]

        raise ValueError("No user messages found in the chat history.")

    async def _get_messages_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        tokenizer_fn: TokenizerFn | None = None,
    ) -> int:
        if not messages:
            return 0
        if isinstance(messages, ChatMessage):
            messages = [messages]

        return await estimate_token_count(
            messages,
            tokenizer_fn=tokenizer_fn,
            message_to_input=self.message_to_input,
        )

    async def _summarize(
        self,
        llm: LLM,
        tokenizer_fn: TokenizerFn | None,
        texts: list[str],
        prompt: BasePromptTemplate | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        if not texts:
            return None

        async def stop_condition(text: str) -> bool:
            if max_tokens is not None:
                token_count = (
                    await async_tokenizer(text, tokenizer_fn=tokenizer_fn)
                    if tokenizer_fn
                    else None
                )
                if token_count is not None:
                    return len(token_count) <= max_tokens

            return False

        workflow = await asyncio.to_thread(
            self.builder.build,
            texts=texts,
            stop_condition_fn=stop_condition,
            timeout=60 * 60,  # 1 hour
        )
        summary = await workflow.run_summary(
            prompt=prompt.format() if prompt else None,
            stream=False,
            empty_response_fallback="-",
        )

        summary_content = "\n".join(block.text for block in summary)
        if not summary_content:
            return None
        return summary_content.strip()

    async def _summarize_obj(
        self,
        llm: LLM,
        output_cls: type[BaseModel],
        texts: list[str],
        prompt: BasePromptTemplate | None = None,
    ) -> BaseModel | None:
        if not texts:
            return None

        content = prompt.format(text="\n".join(texts)) if prompt else ""
        final_prompt = PromptTemplate(template=content)

        result: BaseModel | None = await llm.astructured_predict(
            output_cls=output_cls,
            prompt=final_prompt,
        )
        if not result:
            return None

        return result

    async def _create_chat_batches(
        self,
        chat_history: Sequence[ChatMessage],
        chunk_size: int,
        max_length: int,
        tokenizer_fn: TokenizerFn | None,
    ) -> list[list[ChatMessage]]:
        """Create batches from chat history with async token counting."""

        async def async_chat_generator() -> AsyncIterator[ChatMessage]:
            nonlocal chat_history
            for message in chat_history:
                yield message

        async def token_stop_condition(batch: list[ChatMessage]) -> bool:
            nonlocal max_length
            token_count = await self._get_messages_tokens(
                batch, tokenizer_fn=tokenizer_fn
            )
            return token_count > max_length

        batches: list[list[ChatMessage]] = []
        async for batch in aiter_batch(
            async_chat_generator(), size=chunk_size, stop_condition=token_stop_condition  # type: ignore[call-arg,return-value]
        ):
            batches.append(batch)

        return batches

    async def _summary_left(
        self,
        llm: LLM,
        tokenizer_fn: TokenizerFn | None,
        chat_history: list[ChatMessage],
        max_length: int,
        chunk_size: int = 64,
    ) -> list[ChatMessage]:
        """Summarize left part of the conversation."""
        if not chat_history:
            return []

        current_tokens = await self._get_messages_tokens(
            chat_history, tokenizer_fn=tokenizer_fn
        )
        if current_tokens <= max_length:
            return chat_history

        chat_history = chat_history.copy()

        # 1. Drop the oldest messages until we reach the max length
        chat_history = await self._drop_in_direction(
            chat_history=chat_history,
            direction="left",
            max_length=max_length,
            tokenizer_fn=tokenizer_fn,
            current_tokens=current_tokens,
        )
        if not chat_history:
            return []

        # 2. Summarize the left part of the conversation by chunks
        batches: list[list[ChatMessage]] = await self._create_chat_batches(
            chat_history=chat_history,
            chunk_size=chunk_size,
            max_length=max_length,
            tokenizer_fn=tokenizer_fn,
        )
        adapter: Any = TypeAdapter(list[_SimplifiedTextMessage])
        prompt = await asyncio.to_thread(
            self.prompt_builder_service.create_summary_history_approximately
        )

        summarized_history_tasks: list[Coroutine[Any, Any, BaseModel | None]] = []
        for batch in batches:
            if not batch:
                continue

            texts = [
                await asyncio.to_thread(messages_to_history_str, b) for b in batches
            ]
            summarized_history_tasks.append(
                self._summarize_obj(
                    llm=llm,
                    output_cls=adapter,
                    texts=texts,
                    prompt=prompt,
                )
            )

        results: list[Any] = list(await asyncio.gather(*summarized_history_tasks))
        summaries: list[_SimplifiedTextMessage] = [
            message
            for result in results
            if result and isinstance(result, list)
            for message in result
            if message and isinstance(message, _SimplifiedTextMessage)
        ]

        # 3. Convert summaries to ChatMessage format
        chat_history = [
            ChatMessage(
                role=MessageRole(simplified_summary_message.role),
                content=simplified_summary_message.content or "",
                additional_kwargs={
                    "tldr": "left",
                },
            )
            for simplified_summary_message in summaries
        ]

        return await asyncio.to_thread(
            guarantee_valid_message_sequence, messages=chat_history
        )

    async def _summary_right(
        self,
        llm: LLM,
        tokenizer_fn: TokenizerFn | None,
        chat_history: list[ChatMessage],
        last_user_message: ChatMessage,
        max_length: int,
        aggressive_level: int = 0,
    ) -> list[ChatMessage]:
        """Summarize right part of the conversation."""
        if not chat_history:
            return []

        current_tokens = await self._get_messages_tokens(
            chat_history, tokenizer_fn=tokenizer_fn
        )
        if current_tokens <= max_length:
            return chat_history

        # After a user message, we can find: (user, [assistant, tool]*, assistant?)
        # Assuming that we don't want to summarize the last assistant message,
        # we will summarize pairs of assistant and tool messages.

        # 1. We need to summary pair of assistant and tool messages
        pairs = await asyncio.to_thread(get_assistant_tool_pair_mesages, chat_history)
        if not pairs:
            # If there are no pairs, we return the original chat history
            return chat_history

        # 2. Summarize each pair
        prompt = await asyncio.to_thread(
            self.prompt_builder_service.create_summary_history_in_details,
            user_query=last_user_message.content or "",
            aggressive_level=aggressive_level,
        )

        async def create_summary_task(
            pair: tuple[ChatMessage, ChatMessage],
        ) -> str | None:
            text = await asyncio.to_thread(
                messages_to_history_str, pair, show_index=True
            )
            return await self._summarize(
                llm=llm,
                tokenizer_fn=tokenizer_fn,
                texts=[text],
                prompt=prompt,
                max_tokens=max_length,
            )

        summary_tasks = [create_summary_task(pair) for pair in pairs]
        summaries = await asyncio.gather(*summary_tasks)

        # 3. Replace tool response with summaries
        condense_chat_history = chat_history.copy()
        for (assistant_msg, tool_msg), summary in zip(pairs, summaries, strict=False):
            if summary is None:
                continue

            assistant_index = condense_chat_history.index(assistant_msg)
            tool_index = condense_chat_history.index(tool_msg)
            condense_chat_history[tool_index].blocks = [
                TextBlock(text=summary),
            ]

            # Mark as right-side TLDR
            condense_chat_history[assistant_index].additional_kwargs["tldr"] = "right"
            condense_chat_history[tool_index].additional_kwargs["tldr"] = "right"

            # Remove any non-essential tool keys
            tool_keys = [
                key
                for key in condense_chat_history[tool_index].additional_kwargs
                if not key.startswith("tool")
            ]
            for key in tool_keys:
                if key != "tldr":  # Keep the tldr marker
                    del condense_chat_history[tool_index].additional_kwargs[key]

        return condense_chat_history

    def _drop_tool_messages(
        self,
        chat_history: list[ChatMessage],
    ) -> list[ChatMessage]:
        current_chat_history = chat_history.copy()
        i = 0

        while i < len(current_chat_history):
            if current_chat_history[i].role == "tool":
                # Remove the tool message
                current_chat_history.pop(i)

                # Remove preceding assistant message if it exists
                if i > 0 and current_chat_history[i - 1].role == "assistant":
                    current_chat_history.pop(i - 1)
                    i -= 1
            else:
                i += 1

        return guarantee_valid_message_sequence(current_chat_history)

    def _drop_thinking(
        self,
        chat_history: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Strip thinking from all messages (left side — full removal)."""
        return [
            ChatMessage(
                role=msg.role,
                content=msg.content,
                additional_kwargs={
                    k: v for k, v in msg.additional_kwargs.items() if k != "thinking"
                },
            )
            if "thinking" in msg.additional_kwargs
            else msg
            for msg in chat_history
        ]

    async def _drop_in_direction(
        self,
        chat_history: list[ChatMessage],
        direction: Literal["left", "right"],
        tokenizer_fn: TokenizerFn | None,
        max_length: int,
        current_tokens: int | None = None,
    ) -> list[ChatMessage]:
        if current_tokens is None:
            current_tokens = await self._get_messages_tokens(
                chat_history, tokenizer_fn=tokenizer_fn
            )
        if current_tokens <= max_length:
            return chat_history

        # Split conversation into blocks by user messages
        blocks: list[list[ChatMessage]] = await asyncio.to_thread(
            get_user_blocks, chat_history
        )

        if direction == "right":
            blocks.reverse()

        # Count all blocks in parallel instead of sequentially in a while-loop
        block_token_counts: list[int] = list(
            await asyncio.gather(
                *[
                    self._get_messages_tokens(block, tokenizer_fn=tokenizer_fn)
                    for block in blocks
                ]
            )
        )

        # Drop blocks from the front until we fit within max_length
        drop_until = 0
        for block_tokens in block_token_counts:
            if current_tokens <= max_length:
                break
            current_tokens -= block_tokens
            drop_until += 1

        remaining_blocks = blocks[drop_until:]

        # Flatten remaining blocks back to message list
        result = [msg for block in remaining_blocks for msg in block]

        if direction == "right":
            result.reverse()

        return result

    def _drop_thinking_except_last(
        self,
        chat_history: list[ChatMessage],
    ) -> list[ChatMessage]:
        last_idx = next(
            (
                i
                for i in range(len(chat_history) - 1, -1, -1)
                if "thinking" in chat_history[i].additional_kwargs
            ),
            None,
        )
        if last_idx is None:
            return chat_history
        return self._drop_thinking(chat_history[:last_idx]) + chat_history[last_idx:]

    async def _condense_from_left(
        self,
        llm: LLM,
        tokenizer_fn: TokenizerFn | None,
        chat_history: list[ChatMessage],
        max_length: int,
        left_tokens: int | None = None,
        last_user_tokens: int | None = None,
        right_tokens: int | None = None,
        iteration: int = 0,
    ) -> list[ChatMessage]:
        if iteration > MAX_CONDENSE_ITERATIONS:
            raise ValueError("Maximum number of iterations for condensing exceeded.")

        (
            left_messages,
            last_user_message,
            right_messages,
        ) = await asyncio.to_thread(self._split_conversation, chat_history)

        # Compute per-component token counts if not provided by caller
        if left_tokens is None or last_user_tokens is None or right_tokens is None:
            left_tokens, last_user_tokens, right_tokens = await asyncio.gather(
                self._get_messages_tokens(left_messages, tokenizer_fn=tokenizer_fn),
                self._get_messages_tokens(last_user_message, tokenizer_fn=tokenizer_fn),
                self._get_messages_tokens(right_messages, tokenizer_fn=tokenizer_fn),
            )

        # -1. Drop thinking from left/right, then re-count changed parts in parallel.
        new_left = self._drop_thinking(left_messages)
        new_right = self._drop_thinking(right_messages)
        has_thinking_left = any(
            "thinking" in m.additional_kwargs for m in left_messages
        )
        has_thinking_right = any(
            "thinking" in m.additional_kwargs for m in right_messages
        )
        if has_thinking_left or has_thinking_right:
            recount_tasks = []
            if has_thinking_left:
                recount_tasks.append(
                    self._get_messages_tokens(new_left, tokenizer_fn=tokenizer_fn)
                )
            if has_thinking_right:
                recount_tasks.append(
                    self._get_messages_tokens(new_right, tokenizer_fn=tokenizer_fn)
                )
            recount_results = list(await asyncio.gather(*recount_tasks))
            idx = 0
            if has_thinking_left:
                left_tokens = recount_results[idx]
                idx += 1
            if has_thinking_right:
                right_tokens = recount_results[idx]
        left_messages = new_left
        right_messages = new_right

        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*left_messages, last_user_message, *right_messages]

        # 0. Drop tools from left; only re-count if tools were present
        has_tools = any(m.role == "tool" for m in left_messages)
        potential_left_history = await asyncio.to_thread(
            self._drop_tool_messages, left_messages
        )
        if has_tools:
            new_left_tokens = await self._get_messages_tokens(
                potential_left_history, tokenizer_fn=tokenizer_fn
            )
            current_tokens = new_left_tokens + last_user_tokens + right_tokens
            if current_tokens <= max_length:
                return [*potential_left_history, last_user_message, *right_messages]
            left_messages = potential_left_history
            left_tokens = new_left_tokens

        # 1. Summarize the left part of the conversation
        current_left = await self._summary_left(
            llm=llm,
            tokenizer_fn=tokenizer_fn,
            chat_history=left_messages,
            max_length=max_length,
        )
        left_tokens = await self._get_messages_tokens(
            current_left, tokenizer_fn=tokenizer_fn
        )
        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*current_left, last_user_message, *right_messages]

        # 2. Summarize the right part of the conversation
        current_right = await self._summary_right(
            llm=llm,
            tokenizer_fn=tokenizer_fn,
            chat_history=right_messages,
            last_user_message=last_user_message,
            max_length=max_length,
        )
        right_tokens = await self._get_messages_tokens(
            current_right, tokenizer_fn=tokenizer_fn
        )
        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*current_left, last_user_message, *current_right]

        # 3. Drop messages from the left side until we reach the max length
        # available_left_tokens is arithmetic — no Triton call needed
        available_left_tokens = max_length - last_user_tokens - right_tokens
        current_left = await self._drop_in_direction(
            chat_history=current_left,
            direction="left",
            max_length=available_left_tokens,
            tokenizer_fn=tokenizer_fn,
            current_tokens=left_tokens,
        )

        # 4. Repair the left and right parts of the conversation
        current_left, current_right = await asyncio.gather(
            asyncio.to_thread(repair_without_tools, current_left),
            asyncio.to_thread(
                repair_with_tools, [last_user_message, *current_right], strict=False
            ),
        )
        # current_right now includes last_user_message after repair_with_tools;
        # count both in parallel
        repaired_left_tokens, repaired_right_tokens = await asyncio.gather(
            self._get_messages_tokens(current_left, tokenizer_fn=tokenizer_fn),
            self._get_messages_tokens(current_right, tokenizer_fn=tokenizer_fn),
        )
        current_history = [*current_left, *current_right]
        current_tokens = repaired_left_tokens + repaired_right_tokens

        if current_tokens > max_length:
            return await self._condense_from_left(
                llm=llm,
                tokenizer_fn=tokenizer_fn,
                chat_history=current_history,
                max_length=max_length,
                iteration=iteration + 1,
            )

        return current_history

    async def _condense_from_right(
        self,
        llm: LLM,
        tokenizer_fn: TokenizerFn | None,
        chat_history: list[ChatMessage],
        max_length: int,
        left_tokens: int | None = None,
        last_user_tokens: int | None = None,
        right_tokens: int | None = None,
        iteration: int = 0,
    ) -> list[ChatMessage]:
        if iteration > MAX_CONDENSE_ITERATIONS:
            raise ValueError("Maximum number of iterations for condensing exceeded.")

        (
            left_messages,
            last_user_message,
            right_messages,
        ) = await asyncio.to_thread(self._split_conversation, chat_history)

        # Compute per-component token counts if not provided by caller
        if left_tokens is None or last_user_tokens is None or right_tokens is None:
            left_tokens, last_user_tokens, right_tokens = await asyncio.gather(
                self._get_messages_tokens(left_messages, tokenizer_fn=tokenizer_fn),
                self._get_messages_tokens(last_user_message, tokenizer_fn=tokenizer_fn),
                self._get_messages_tokens(right_messages, tokenizer_fn=tokenizer_fn),
            )

        # 0a. Drop thinking from right (except last occurrence) and from left;
        #     re-count both in parallel since either may have changed
        new_right = self._drop_thinking_except_last(right_messages)
        new_left = self._drop_thinking(left_messages)
        new_left_tokens, new_right_tokens = await asyncio.gather(
            self._get_messages_tokens(new_left, tokenizer_fn=tokenizer_fn),
            self._get_messages_tokens(new_right, tokenizer_fn=tokenizer_fn),
        )
        left_messages = new_left
        right_messages = new_right
        left_tokens = new_left_tokens
        right_tokens = new_right_tokens
        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*left_messages, last_user_message, *right_messages]

        # 0b. Drop remaining thinking from right (left already fully stripped)
        new_right = self._drop_thinking(right_messages)
        has_remaining_thinking = any(
            "thinking" in m.additional_kwargs for m in right_messages
        )
        if has_remaining_thinking:
            right_tokens = await self._get_messages_tokens(
                new_right, tokenizer_fn=tokenizer_fn
            )
        right_messages = new_right
        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*left_messages, last_user_message, *right_messages]

        # 1. Summarize the right part of the conversation
        current_right = await self._summary_right(
            llm=llm,
            tokenizer_fn=tokenizer_fn,
            chat_history=right_messages,
            last_user_message=last_user_message,
            max_length=max_length,
            aggressive_level=iteration,
        )
        right_tokens = await self._get_messages_tokens(
            current_right, tokenizer_fn=tokenizer_fn
        )
        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*left_messages, last_user_message, *current_right]

        # 3. Summarize the left part of the conversation
        current_left = await self._summary_left(
            llm=llm,
            tokenizer_fn=tokenizer_fn,
            chat_history=left_messages,
            max_length=max_length,
        )
        left_tokens = await self._get_messages_tokens(
            current_left, tokenizer_fn=tokenizer_fn
        )
        current_tokens = left_tokens + last_user_tokens + right_tokens
        if current_tokens <= max_length:
            return [*current_left, last_user_message, *current_right]

        # 4. Drop messages from the left side until we reach the max length
        # right_side is arithmetic — no Triton call needed
        right_side = last_user_tokens + right_tokens
        if right_side >= max_length:
            current_history = [*current_left, last_user_message, *current_right]
            return await self._condense_from_right(
                llm=llm,
                tokenizer_fn=tokenizer_fn,
                chat_history=current_history,
                max_length=max_length,
                iteration=iteration + 1,
            )

        available_left_tokens = max_length - right_side
        current_left = await self._drop_in_direction(
            chat_history=current_left,
            direction="left",
            max_length=available_left_tokens,
            tokenizer_fn=tokenizer_fn,
            current_tokens=left_tokens,
        )

        # 5. Repair the left and right parts of the conversation
        current_left, current_right = await asyncio.gather(
            asyncio.to_thread(repair_without_tools, current_left),
            asyncio.to_thread(
                repair_with_tools, [last_user_message, *current_right], strict=False
            ),
        )
        # current_right now includes last_user_message; count both in parallel
        repaired_left_tokens, repaired_right_tokens = await asyncio.gather(
            self._get_messages_tokens(current_left, tokenizer_fn=tokenizer_fn),
            self._get_messages_tokens(current_right, tokenizer_fn=tokenizer_fn),
        )
        current_history = [*current_left, *current_right]
        current_tokens = repaired_left_tokens + repaired_right_tokens

        if current_tokens > max_length:
            return await self._condense_from_right(
                llm=llm,
                tokenizer_fn=tokenizer_fn,
                chat_history=current_history,
                max_length=max_length,
                iteration=iteration + 1,
            )

        return current_history

    async def get_memory(
        self,
        chat_history: list[ChatMessage],
        max_length: int | None = None,
        **kwargs: dict[str, Any],
    ) -> list[ChatMessage]:
        """Get memory with summarization for overflowing context."""
        if not chat_history or max_length is None or max_length <= 0:
            return chat_history

        (
            left_messages,
            last_user_message,
            right_messages,
        ) = await asyncio.to_thread(self._split_conversation, chat_history.copy())

        # Select LLM and tokenizer
        model_id: str | None = kwargs.get("model_id")  # type: ignore[assignment]
        llm = self.llm_component.get_llm(model_id)
        tokenizer = self.llm_component.get_tokenizer(model_id)

        left_tokens, last_user_message_tokens, right_tokens = await asyncio.gather(
            self._get_messages_tokens(left_messages, tokenizer_fn=tokenizer),
            self._get_messages_tokens(last_user_message, tokenizer_fn=tokenizer),
            self._get_messages_tokens(right_messages, tokenizer_fn=tokenizer),
        )

        if last_user_message_tokens > max_length:
            raise ValueError(
                "The last user message exceeds the maximum length allowed."
            )
        if left_tokens + last_user_message_tokens + right_tokens <= max_length:
            return chat_history

        # Decide in which direction we will go
        left_token_percentage = left_tokens / (
            left_tokens + right_tokens + last_user_message_tokens
        )
        right_token_percentage = right_tokens / (
            left_tokens + right_tokens + last_user_message_tokens
        )
        diff_tokens = right_token_percentage - left_token_percentage

        # If there exists a significant difference in left token distribution,
        # we will perform condensation from the left side
        if diff_tokens <= 0.1:
            chat_history = await self._condense_from_left(
                llm=llm,
                tokenizer_fn=tokenizer,
                chat_history=chat_history,
                max_length=max_length,
                left_tokens=left_tokens,
                last_user_tokens=last_user_message_tokens,
                right_tokens=right_tokens,
            )
        else:
            chat_history = await self._condense_from_right(
                llm=llm,
                tokenizer_fn=tokenizer,
                chat_history=chat_history,
                max_length=max_length,
                left_tokens=left_tokens,
                last_user_tokens=last_user_message_tokens,
                right_tokens=right_tokens,
            )

        # After condensation, we ensure that the last user message is preserved
        # and the chat history is within the maximum length
        if not any(msg.role == "user" for msg in chat_history):
            raise ValueError("No user messages found after condensation.")

        if not last_user_message:
            raise ValueError("No last user message found after condensation.")

        tokens = await self._get_messages_tokens(chat_history, tokenizer_fn=tokenizer)
        if tokens > max_length:
            raise ValueError(
                "Condensed chat history exceeds maximum length after applying condensation strategy."
            )

        return chat_history
