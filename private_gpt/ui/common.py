from pydantic import BaseModel
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.constants import UPLOAD_DIR
from pathlib import Path
class Source(BaseModel):
    file: str
    page: str
    text: str
    page_link: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> set["Source"]:
        curated_sources = set()

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"
            page_link = str(Path(f"{UPLOAD_DIR}/{file_name}#page={page_label}"))

            source = Source(file=file_name, page=page_label, text=chunk.text, page_link=page_link)
            curated_sources.add(source)

        return curated_sources