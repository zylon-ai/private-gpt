import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.citations.utils import (
    extract_citations_by_original_text,
)
from private_gpt.events.models import (
    Event,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextDelta,
    ThinkingDelta,
)

if TYPE_CHECKING:
    from private_gpt.components.engines.citations.types import Citation, Document


class ExtractCitationInterceptor(ChatResponseLoopInterceptor):
    def __init__(self) -> None:
        self._send_text: str = ""
        self._send_citations: list[Citation] = []
        self._current_text: str = ""
        self._documents: list[Document] = []
        self._citation_indices: dict[str, int] = {}

    async def on_iteration_start(self, context: ChatInterceptorContext) -> None:
        self._send_text = ""
        self._send_citations = []
        self._current_text = ""

    async def on_iteration_end(self, context: ChatInterceptorContext) -> None:
        # Mutate the context state to include the final citations for this iteration
        if self._send_citations:
            new_state = context.state.model_copy(deep=True)
            existing = new_state.input.request.citation.citations or []
            additions = [c for c in self._send_citations if c not in existing]
            new_state.input.request.citation.citations = [*existing, *additions]
            context.set_state(new_state)

        # Reset documents & citation indices
        self._documents = []
        self._citation_indices = {}

    async def intercept_event(
        self,
        event: Event,
        context: ChatInterceptorContext,
    ) -> Event | None:
        citations_is_enabled = context.state.input.request.citation.enabled
        documents = self._documents or context.state.input.context_stack.all_documents()

        if not (citations_is_enabled and bool(documents)):
            return event

        if isinstance(event, RawContentBlockStartEvent):
            self._send_text = ""
            self._send_citations = []
            self._current_text = ""
            return event

        if not isinstance(event, RawContentBlockDeltaEvent) or not event.delta:
            return event

        if isinstance(event.delta, TextDelta):
            delta_text = event.delta.text or ""
            self._current_text += delta_text
        elif isinstance(event.delta, ThinkingDelta):
            delta_text = event.delta.thinking or ""
            self._current_text += delta_text
        else:
            return event

        if not self._citation_indices:
            raw = context.state.input.request.citation.citations or []
            self._citation_indices = {
                c.source_id: int(c.value["index"])
                for c in raw
                if c.source_id
                and c.value is not None
                and c.value.get("index") is not None
            }

        (
            cleaned_text,
            current_citations,
            self._citation_indices,
        ) = await asyncio.to_thread(
            extract_citations_by_original_text,
            text=self._current_text,
            documents=documents,
            citation_indices=self._citation_indices,
        )

        if not cleaned_text:
            return None

        new_delta_text = cleaned_text[len(self._send_text) :]
        new_citations = current_citations[len(self._send_citations) :]

        if not new_delta_text and not new_citations:
            return None

        if isinstance(event.delta, TextDelta):
            event.delta = TextDelta.from_citations(new_delta_text, new_citations)
        else:
            event.delta = ThinkingDelta.from_citations(new_delta_text, new_citations)

        self._send_text = cleaned_text
        self._send_citations.extend(new_citations)

        return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "ExtractCitationInterceptor":
        # Return a new instance with the same logic but reset state
        return ExtractCitationInterceptor()
