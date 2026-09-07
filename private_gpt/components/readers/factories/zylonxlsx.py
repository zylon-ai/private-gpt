from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory


class ZylonXlsxReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension

        from private_gpt.components.readers.zylonxlsx.zylonxlsx_reader import (
            ZylonXlsxReader,
        )

        return ZylonXlsxReader()
