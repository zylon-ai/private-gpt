import asyncio
from abc import abstractmethod
from collections.abc import AsyncIterable
from typing import Any

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import BaseComponent, BaseNode
from pydantic import ConfigDict

from private_gpt.components.ingest.utils import FileInfo


class IngestionReader(BaseComponent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def lazy_load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterable[BaseNode]:
        """Load data from the input directory lazily."""
        pass


class LlamaIndexReaderAdapter(IngestionReader):

    _reader: BaseReader

    def __init__(self, reader: BaseReader) -> None:
        self._reader = reader

    @classmethod
    def from_reader(cls, reader: BaseReader) -> "LlamaIndexReaderAdapter":
        return cls(reader)

    async def lazy_load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterable[Document]:
        del execute_transformations, args

        file_data = file_info.file_data
        extra_info = extra_info or {}

        def has_kwargs() -> bool:
            import inspect

            sig = inspect.signature(self._reader.lazy_load_data)
            params = sig.parameters.values()
            return any(True for p in params if p.kind == p.VAR_KEYWORD)

        documents = await asyncio.to_thread(
            lambda: list(
                self._reader.lazy_load_data(file_data, extra_info, **load_kwargs)
                if has_kwargs()
                else self._reader.lazy_load_data(file_data, extra_info)
            )
        )
        for document in documents:
            yield document
