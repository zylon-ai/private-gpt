import logging
from collections.abc import Sequence
from typing import Any, Literal

import jinja2
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    MessageRole,
)
from llama_index.core.tools import BaseTool

from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.llm.prompt_styles.prompt_style_base import (
    PromptData,
    PromptStyleBase,
)
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.components.llm.utils import format_hf_conversation

logger = logging.getLogger(__name__)

ChatTemplateContentFormat = Literal["string", "openai"]

_FALLBACK_USER_MESSAGE = {"role": "user", "content": ""}


class ChatTemplatePromptStyle(PromptStyleBase):
    def __init__(
        self,
        tokenizer: TokenizerBase | None = None,
        content_format: ChatTemplateContentFormat = "string",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Tokenizer is required and must support apply_chat_template: {tokenizer}"
            )
        self._tokenizer = tokenizer
        self._content_format = content_format

    def _messages_to_prompt(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> PromptData:
        reasoning_effort = reasoning_effort or ReasoningEffort.NONE
        continue_final_message = (
            bool(messages) and messages[-1].role == MessageRole.ASSISTANT
        )
        conversation = self._to_hf_messages(messages)
        prompt = self._apply_chat_template(
            conversation=conversation,
            tokenize=tokenize,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
            reasoning_effort=reasoning_effort,
        )
        images = [
            b.resolve_image()
            for m in messages
            for b in m.blocks
            if isinstance(b, ImageBlock) and m.role == MessageRole.USER
        ]
        audios = [
            b.resolve_audio()
            for m in messages
            for b in m.blocks
            if isinstance(b, AudioBlock) and m.role == MessageRole.USER
        ]
        return PromptData(
            prompt=prompt if isinstance(prompt, str) else None,
            token_ids=None if isinstance(prompt, str) else prompt,
            images=images or None,
            audios=audios or None,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

    def _completion_to_prompt(self, completion: str, **kwargs: Any) -> PromptData:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)], **kwargs
        )

    def support_chat_template(self, tokenizer: Any) -> bool:
        return tokenizer is not None or hasattr(tokenizer, "apply_chat_template")

    def _apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
        continue_final_message: bool,
        reasoning_effort: ReasoningEffort,
    ) -> list[int] | str:
        if not self._tokenizer.support_chat_template(self._tokenizer):
            return format_hf_conversation(
                conversation, add_generation_prompt=add_generation_prompt
            )

        try:
            return self._tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                enable_thinking=reasoning_effort.is_thinking_enabled,
                reasoning_effort=reasoning_effort.value,
            )
        except jinja2.exceptions.TemplateError:
            sanitized = self._sanitize_conversation(conversation)
            return self._tokenizer.apply_chat_template(
                conversation=sanitized,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                enable_thinking=reasoning_effort.is_thinking_enabled,
                reasoning_effort=reasoning_effort.value,
            )
        except NotImplementedError:
            return format_hf_conversation(
                conversation, add_generation_prompt=add_generation_prompt
            )

    @staticmethod
    def _sanitize_conversation(
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Drop non-user/assistant/system messages that some templates reject
        allowed_roles = {"system", "user", "assistant"}
        sanitized = [m for m in conversation if m.get("role") in allowed_roles]

        # Ensure there is at least one user message
        has_user = any(m.get("role") == "user" for m in sanitized)
        if not has_user:
            # Insert a blank user message before the first assistant turn, or append
            insert_at = next(
                (i for i, m in enumerate(sanitized) if m.get("role") == "assistant"),
                len(sanitized),
            )
            sanitized.insert(insert_at, _FALLBACK_USER_MESSAGE)

        return sanitized

    def _to_hf_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        openai_dicts: list[dict[str, Any]] = []

        for message in messages:
            msg_dict: dict[str, Any] = {
                **message.additional_kwargs,
                "role": message.role.value,
                "content": message.content or "",
            }

            openai_dicts.append(msg_dict)

        if self._content_format == "string":
            return [
                {**msg, "content": self._collapse_to_string(msg.get("content"))}
                for msg in openai_dicts
            ]
        return openai_dicts

    @staticmethod
    def _collapse_to_string(content: Any) -> str:
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        return str(content)
