import re
from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

from injector import singleton

from private_gpt.chat.input_models import (
    CompletionInput,
    CompletionOutput,
    MessageInput,
)
from private_gpt.events.models import TextBlock
from private_gpt.server.chat.chat_models import ChatBody


@singleton
class CompletionService:
    """Service for adapting legacy completion payloads into chat requests."""

    @staticmethod
    def _parse_prompt(prompt: str) -> list[MessageInput]:
        pattern = re.compile(r"(?:\A|\n\n)(Human|Assistant):")
        matches = list(pattern.finditer(prompt))
        if not matches:
            return [MessageInput(role="user", content=prompt)]

        messages: list[MessageInput] = []
        for i, match in enumerate(matches):
            role: Literal["user", "assistant"] = (
                "user" if match.group(1) == "Human" else "assistant"
            )
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
            content = prompt[start:end].strip()

            if role == "assistant" and i == len(matches) - 1 and not content:
                continue

            messages.append(MessageInput(role=role, content=content))

        return messages

    def to_chat_body(self, body: CompletionInput) -> ChatBody:
        return ChatBody(
            model=body.model,
            stream=body.stream,
            messages=self._parse_prompt(body.prompt),
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            max_tokens=body.max_tokens_to_sample,
        )

    @staticmethod
    def to_completion_output(
        completion: str,
        stop_reason: str | None,
        model: str,
    ) -> CompletionOutput:
        return CompletionOutput(
            id=f"compl_{uuid4().hex}",
            type="completion",
            completion=completion,
            stop_reason=stop_reason or "end_turn",
            model=model,
        )

    @staticmethod
    def extract_text_from_content(content: Sequence[object]) -> str:
        return "".join(block.text for block in content if isinstance(block, TextBlock))
