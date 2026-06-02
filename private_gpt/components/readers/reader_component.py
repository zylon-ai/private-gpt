import logging
from typing import Any

from injector import inject, singleton
from llama_index.core.schema import BaseNode

from private_gpt.components.ingest.utils import FileInfo
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory
from private_gpt.components.readers.factories.factory import ReaderFactoryRegistry
from private_gpt.components.readers.registry import ReaderRegistry
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class ReaderComponent:
    @inject
    def __init__(
        self,
        settings: Settings,
        reader_registry: ReaderRegistry,
        reader_factory_registry: ReaderFactoryRegistry,
    ) -> None:
        self.settings = settings
        self.registry = reader_registry
        self.factory_registry = reader_factory_registry

    def get_reader_by_extension(self, extension: str) -> IngestionReader:
        reader_names = self.get_reader_names(extension=extension)
        if not reader_names:
            raise ValueError(f"No reader registered for extension: {extension}")
        return self.get_reader(reader_names[0], extension)

    def get_reader(
        self,
        name: str | None = None,
        extension: str | None = None,
    ) -> IngestionReader:
        reader_names = self.get_reader_names(name=name, extension=extension)
        if not reader_names:
            raise ValueError(f"No reader available for extension: {extension}")
        target_name = reader_names[0]
        factory = self.factory_registry.get_factory(target_name)
        return factory.create_reader(extension)

    def get_registered_reader_name(self, extension: str | None) -> str | None:
        return self.registry.get_reader_name(extension)

    def get_reader_names(
        self,
        name: str | None = None,
        extension: str | None = None,
    ) -> list[str]:
        if name and name != "auto":
            return [name]

        reader_names = self.registry.get_reader_names(extension)
        if self.settings.data.reader in reader_names:
            reader_names = [self.settings.data.reader]
        return reader_names or ["text"]

    async def load_data(
        self,
        file_info: FileInfo,
        extra_info: dict[str, Any] | None = None,
        **load_kwargs: Any,
    ) -> list[BaseNode]:
        reader_names = self.get_reader_names(extension=file_info.extension)
        last_exception: Exception | None = None

        for reader_name in reader_names:
            try:
                loader = self.get_reader(reader_name, file_info.extension)
            except ImportError as e:
                last_exception = e
                logger.warning(
                    "Reader '%s' could not be loaded for file '%s': %s",
                    reader_name,
                    file_info.file_name,
                    e,
                )
                continue

            nodes: list[BaseNode] = []
            async for node in loader.lazy_load_data(
                file_info,
                extra_info=extra_info,
                **load_kwargs,
            ):
                nodes.append(node)
            if nodes:
                return nodes
            logger.info(
                "Reader '%s' returned no nodes for file: %s",
                reader_name,
                file_info.file_name,
            )

        if last_exception is not None:
            available = ", ".join(reader_names)
            raise RuntimeError(
                f"Failed to load file '{file_info.file_name}' with readers: {available}"
            ) from last_exception

        return []

    def register_reader_factory(self, name: str, factory: ReaderFactory) -> None:
        self.factory_registry.register_factory(name, factory)

    def unregister_reader_factory(self, name: str) -> None:
        self.factory_registry.unregister_factory(name)

    def register_extension_reader(self, extension: str, reader_name: str) -> None:
        self.factory_registry.get_factory(reader_name)
        self.registry.register_extension_reader(extension, reader_name)

    def register_extension_readers(
        self,
        extension: str,
        reader_names: list[str],
    ) -> None:
        for reader_name in reader_names:
            self.factory_registry.get_factory(reader_name)
        self.registry.register_extension_readers(extension, reader_names)

    def unregister_extension_reader(self, extension: str) -> None:
        self.registry.unregister_extension_reader(extension)
