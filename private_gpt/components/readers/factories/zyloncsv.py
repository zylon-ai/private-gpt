from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory


class ZylonCsvReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension

        from private_gpt.components.readers.zyloncsv.zyloncsv_reader import (
            ZylonCsvReader,
        )

        return ZylonCsvReader()
