import asyncio
import logging
import time
from abc import abstractmethod
from collections.abc import AsyncIterable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import arun_transformations
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import BaseComponent, BaseNode, TransformComponent
from pydantic import ConfigDict

from private_gpt.components.ingest.utils import FileInfo
from private_gpt.settings.settings import settings

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


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

    @contextmanager
    def _timed_phase(self, phase: str, file_name: str | None) -> Any:
        """Measure and log the time taken by a reader phase (e.g. parsing).

        Wrap the raw document-extraction step of a reader's ``lazy_load_data``
        so timings are comparable across reader implementations.
        """
        reader_name = self.__class__.__name__
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            logger.debug(
                "[TIMING] %s %s file: %s %.3fs",
                reader_name,
                phase,
                file_name,
                elapsed,
            )

    async def _run_transformations_with_timing(
        self,
        nodes: Sequence[BaseNode],
        transformations: Iterable[TransformComponent],
        file_name: str | None,
    ) -> list[BaseNode]:
        """Run transformations, logging the time taken by each when debug logging is on."""
        if not logger.isEnabledFor(logging.DEBUG):
            return list(await arun_transformations(list(nodes), list(transformations)))

        reader_name = self.__class__.__name__
        result: list[BaseNode] = list(nodes)
        timings: list[tuple[str, float, int]] = []

        for transform in transformations:
            t0 = time.perf_counter()
            result = list(await arun_transformations(result, [transform]))
            elapsed = time.perf_counter() - t0
            timings.append((transform.__class__.__name__, elapsed, len(result)))
            logger.debug(
                "[TIMING] %s %-40s %8.3fs  (nodos resultantes: %d)",
                reader_name,
                transform.__class__.__name__,
                elapsed,
                len(result),
            )

        total = sum(t for _, t, _ in timings)
        if total:
            logger.debug(
                "[TIMING] %s TOTAL %s: %.3fs -> desglose: %s",
                reader_name,
                file_name,
                total,
                {name: f"{t:.3f}s ({t / total * 100:.1f}%)" for name, t, _ in timings},
            )

        return result


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
