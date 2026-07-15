import asyncio
import json
import logging
import typing
from collections.abc import Generator, Sequence
from typing import Any

from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.program.utils import FlexibleModel
from llama_index.core.tools import BaseTool
from llama_index.core.types import Model
from pydantic import BaseModel, TypeAdapter

from private_gpt.components.llm.models import ReasoningEffort

logger = logging.getLogger(__name__)


class StructuredChatMixin:
    """Mixin that adds structured-output predict/chat methods to an LLM class."""

    @staticmethod
    def _parse_partial_json(
        text: str,
        output_cls: type[Model],
        flags: Any,
    ) -> "Model | FlexibleModel | None":
        from partial_json_parser.core.exceptions import MalformedJSON  # type: ignore
        from partial_json_parser.core.options import Allow  # type: ignore  # noqa: F401

        from private_gpt.components.llm.utils import partial_json_loads

        try:
            partial_obj = partial_json_loads(text, flags=flags)[0]
            text = json.dumps(partial_obj, ensure_ascii=False)
        except MalformedJSON:
            logger.debug("Not enough tokens to parse into JSON yet")
        except Exception as e:
            logger.debug("Error parsing JSON: %s", e)

        if not text or text.strip() == "" or text.strip() == "{}":
            return None

        output: Model | FlexibleModel | None = None
        try:
            if isinstance(output_cls, TypeAdapter):
                output = output_cls.validate_json(text)
            elif isinstance(output_cls, BaseModel | type(BaseModel)):
                output = output_cls.model_validate_json(text)
        except Exception:
            try:
                output = FlexibleModel.model_validate_json(text)
            except ValueError:
                output = None

        return output

    def structured_predict(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Model:
        items: list[Model | FlexibleModel] = list(
            self.stream_structured_predict(
                output_cls=output_cls,
                prompt=prompt,
                llm_kwargs=llm_kwargs,
                **prompt_args,
            )
        )
        last_item: Model | FlexibleModel = items[-1]
        if isinstance(last_item, FlexibleModel):
            raise ValueError(
                "Last item is a FlexibleModel, expected a specific output_cls."
            )
        return last_item  # type: ignore[return-value]

    def stream_structured_predict(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Generator[Model | FlexibleModel, None, None]:
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=prompt.format(**prompt_args),
            )
        ]
        return self.stream_structured_chat(
            output_cls=output_cls,
            messages=messages,
            **(llm_kwargs or {}),
        )

    def structured_chat(
        self,
        output_cls: type[Model],
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        allow_flexible: bool = False,
        **kwargs: Any,
    ) -> "Model | FlexibleModel":
        items: list[Model | FlexibleModel] = list(
            self.stream_structured_chat(
                output_cls=output_cls,
                messages=messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                **kwargs,
            )
        )
        last_item: Model | FlexibleModel = items[-1]
        if isinstance(last_item, FlexibleModel) and not allow_flexible:
            raise ValueError(
                "Last item is a FlexibleModel, expected a specific output_cls."
            )
        return last_item

    def stream_structured_chat(
        self,
        output_cls: type[Model],
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> Generator[Model | FlexibleModel, None, None]:
        from partial_json_parser.core.options import Allow  # type: ignore

        from private_gpt.components.llm.custom.base import StructuredOutputsParams

        structured_outputs = StructuredOutputsParams.from_optional(
            output_cls=output_cls
        )
        flags = Allow.ALL & ~Allow.STR

        for response in self.stream_chat(  # type: ignore[attr-defined]
            messages=messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            structured_outputs=structured_outputs,
            **kwargs,
        ):
            if not response.message.content:
                continue
            output = self._parse_partial_json(
                response.message.content, output_cls, flags
            )
            if output is not None:
                yield output

    async def astructured_predict(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Model:
        items: list[Model | FlexibleModel] = []
        async for item in await self.astream_structured_predict(
            output_cls=output_cls,
            prompt=prompt,
            llm_kwargs=llm_kwargs,
            **prompt_args,
        ):
            items.append(item)
        if not items:
            raise ValueError("No items returned from astream_structured_predict")
        last_item: Model | FlexibleModel = items[-1]
        if isinstance(last_item, FlexibleModel):
            raise ValueError(
                "Last item is a FlexibleModel, expected a specific output_cls."
            )
        return last_item  # type: ignore[return-value]

    async def astream_structured_predict(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> typing.AsyncGenerator[Model | FlexibleModel, None]:
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=await asyncio.to_thread(prompt.format, **prompt_args),
            )
        ]
        return await self.astream_structured_chat(
            output_cls=output_cls,
            messages=messages,
            **(llm_kwargs or {}),
        )

    async def astructured_chat(
        self,
        output_cls: type[Model],
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        allow_flexible: bool = False,
        **kwargs: Any,
    ) -> "Model | FlexibleModel":
        items: list[Model | FlexibleModel] = []
        async for item in await self.astream_structured_chat(
            output_cls=output_cls,
            messages=messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            **kwargs,
        ):
            items.append(item)
        if not items:
            raise ValueError("No items returned from astream_structured_chat")
        last_item: Model | FlexibleModel = items[-1]
        if isinstance(last_item, FlexibleModel) and not allow_flexible:
            raise ValueError(
                "Last item is a FlexibleModel, expected a specific output_cls."
            )
        return last_item

    async def astream_structured_chat(
        self,
        output_cls: type[Model],
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> typing.AsyncGenerator[Model | FlexibleModel, None]:
        from partial_json_parser.core.options import Allow  # type: ignore

        from private_gpt.components.llm.custom.base import StructuredOutputsParams

        structured_outputs = await asyncio.to_thread(
            StructuredOutputsParams.from_optional,
            output_cls=output_cls,
        )
        flags = Allow.ALL & ~Allow.STR

        async def gen() -> typing.AsyncGenerator[Model | FlexibleModel, None]:
            async for response in await self.astream_chat(  # type: ignore[attr-defined]
                messages=messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                structured_outputs=structured_outputs,
                **kwargs,
            ):
                if not response.message.content:
                    continue
                output = await asyncio.to_thread(
                    self._parse_partial_json,
                    response.message.content,
                    output_cls,
                    flags,
                )
                if output is not None:
                    yield typing.cast("Model | FlexibleModel", output)

        return gen()
