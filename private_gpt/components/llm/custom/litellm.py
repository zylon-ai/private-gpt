"""LiteLLM provider — routes to 100+ LLM providers via litellm.completion().

Supports OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, Gemini, Cohere,
Mistral, Groq, Together AI, Ollama, and more.

Provider API keys are read from environment variables automatically
(OPENAI_API_KEY, ANTHROPIC_API_KEY, AWS_ACCESS_KEY_ID, GEMINI_API_KEY, etc.).

Model names use LiteLLM format: "provider/model-name", e.g.:
    openai/gpt-4o, anthropic/claude-sonnet-4-20250514,
    bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0, gemini/gemini-2.5-flash

See https://docs.litellm.ai/docs/providers for the full list.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import (
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponseGen,
)

logger = logging.getLogger(__name__)


class LiteLLMCustomLLM(CustomLLM):
    """LiteLLM provider for PrivateGPT.

    Uses litellm.completion() directly to route to 100+ LLM providers.
    """

    model: str = Field(description="LiteLLM model name, e.g. 'openai/gpt-4o'.")
    temperature: float = Field(0.1, description="Sampling temperature.")
    max_new_tokens: int = Field(256, description="Maximum tokens to generate.")
    context_window: int = Field(3900, description="Maximum context tokens.")
    request_timeout: float = Field(120.0, description="Request timeout in seconds.")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            timeout=self.request_timeout,
            drop_params=True,
        )
        text = response.choices[0].message.content or ""
        return CompletionResponse(text=text, raw=response.model_dump())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        import litellm

        def gen():
            text = ""
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                timeout=self.request_timeout,
                stream=True,
                drop_params=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                text += content
                yield CompletionResponse(delta=content, text=text, raw=chunk.model_dump())

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        import litellm

        litellm_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]
        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            timeout=self.request_timeout,
            drop_params=True,
        )
        text = response.choices[0].message.content or ""
        completion_response = CompletionResponse(text=text, raw=response.model_dump())
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        import litellm

        litellm_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

        def gen():
            text = ""
            response = litellm.completion(
                model=self.model,
                messages=litellm_messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                timeout=self.request_timeout,
                stream=True,
                drop_params=True,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                text += content
                yield CompletionResponse(delta=content, text=text, raw=chunk.model_dump())

        return stream_completion_response_to_chat_response(gen())
