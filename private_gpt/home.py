"""This file should be imported only and only if you want to run the UI locally."""
from fastapi import Request
from fastapi.responses import StreamingResponse
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, Depends, Request, FastAPI, Body
from fastapi.responses import JSONResponse
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg
from private_gpt.ui.common import Source

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "\n\n Sources: \n"

MODES = ["Query Docs", "Search in Docs", "LLM Chat"]
home_router = APIRouter(prefix="/v1")



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

        # Initialize system prompt based on default mode
        self.mode = MODES[0]
        self._system_prompt = self._get_default_system_prompt(self.mode)

    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n".join(
                    f'<a href="{source.page_link}" target="_blank" rel="noopener noreferrer">{index}. {source.file} (page {source.page})</a>'
                    for index, source in enumerate(cur_sources, start=1)
                )
                full_response += sources_text
                print(full_response)
            yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(
                                content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                # Remove from history content the Sources information
                                content=interaction[1].split(
                                    SOURCES_SEPARATOR)[0],
                                role=MessageRole.ASSISTANT,
                            ),
                        ]
                        for interaction in history
                    ]
                )
            )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
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

                yield "\n\n\n".join(
                    f"{index}. **{source.file} (page {source.page})**\n"
                    f" (link: [{source.page_link}]({source.page_link}))\n{source.text}"
                    for index, source in enumerate(sources, start=1)
                )

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case "Query Docs":
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case "LLM Chat":
                p = settings().ui.default_chat_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_current_mode(self, mode: str) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
    
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



import json

DEFAULT_MODE = MODES[0]
@home_router.post("/chat")
async def chat_endpoint(request: Request, message: str = Body(...), mode: str = Body(DEFAULT_MODE)):
    home_instance = request.state.injector.get(Home)
    history = []
    print("The message is: ", message)
    print("The mode is: ", mode)
    responses = home_instance._chat(message, history, mode)
    return StreamingResponse(content=responses, media_type='text/event-stream')





    # text = (
    #     "To run the Celery worker based on the provided context, you can follow these steps: "
    #     "1. First, make sure you have Celery installed in your project."
    #     "2. Create a Celery instance and configure it with your settings."
    #     "3. Define Celery tasks that will be executed by the worker."
    #     "4. Start the Celery worker using the configured Celery instance."
    #     "5. Your Celery worker is now running and ready to process tasks."
    # )
    # import time
    # async def generate_stream():
    #     for i in range(len(text)):
    #         yield text[:i+1]  # Sending part of the text in each iteration
    #         time.sleep(0.1)  # Simulating some processing time

    # Return the responses as a StreamingResponse
    # return StreamingResponse(content=responses, media_type="application/json")


