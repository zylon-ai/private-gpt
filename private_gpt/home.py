"""This file should be imported only and only if you want to run the UI locally."""
from private_gpt.users.core import security
from private_gpt.users.api import deps
from private_gpt.users import crud, models, schemas
import time
from fastapi import File, Request, UploadFile
from fastapi.responses import StreamingResponse
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Literal

from fastapi import APIRouter, Depends, Request, FastAPI, Body, status, HTTPException, Security
from fastapi.responses import JSONResponse
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg
from private_gpt.ui.common import Source
from private_gpt.constants import UPLOAD_DIR



logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
SOURCES_SEPARATOR = "\n Sources: \n"

MODES = ["Query Docs", "Search in Docs", "LLM Chat"]
DEFAULT_MODE = MODES[0]

home_router = APIRouter(prefix="/v1", tags=["Chat"])

class ListFilesResponse(BaseModel):
    uploaded_files: List[str]

class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@singleton
class Home:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service

        self.mode = MODES[0]
        self._history = []

    def _chat(self, message: str, mode: str, *_: Any) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            print("THE STREAM: ",)
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
               
                sources_text = "\n".join(
                    f'<a href="{source.page_link}" target="_blank" rel="noopener noreferrer">{index}. {source.file} (page {source.page})</a>'
                    for index, source in enumerate(cur_sources, start=1)
                )
                full_response += sources_text
            yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(
                                content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                content=interaction[1].split(
                                    SOURCES_SEPARATOR)[0],
                                role=MessageRole.ASSISTANT,
                            ),
                        ]
                        for interaction in self._history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        self._history.append([message, ""])
        all_messages = [*build_history(), new_message]
        match mode:
            case "Query Docs":
                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                )
                yield from yield_deltas(query_stream)
            case "LLM Chat":
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case "Search in Docs":
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n".join(
                    f"{index}. **{source.file} (page {source.page})**\n"
                    f" (link: [{source.page_link}]({source.page_link}))\n{source.text}"
                    for index, source in enumerate(sources, start=1)
                )

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]
        self._ingest_service.bulk_ingest(
            [(str(path.name), path) for path in paths])


home_instance = global_injector.get(Home)
def get_home_instance(request: Request) -> Home:
    home_instance = request.state.injector.get(Home)
    return home_instance


@home_router.post("/chat")
async def chat_endpoint(
    home_instance: Home = Depends(get_home_instance), 
    message: str = Body(...), mode: str = Body(DEFAULT_MODE),
    current_user: models.User = Security(
        deps.get_current_user,
    )
):
    response = home_instance._chat(message=message, mode=mode)
    return StreamingResponse(
        response,
        media_type='text/event-stream'
    )


@home_router.get("/list_files")
async def list_files(
    home_instance: Home = Depends(get_home_instance),  
    current_user: models.User = Security(
        deps.get_current_user,
)) -> dict:
    """
    List all uploaded files.
    """
    uploaded_files = home_instance._list_ingested_files()
    return {"uploaded_files": uploaded_files}

