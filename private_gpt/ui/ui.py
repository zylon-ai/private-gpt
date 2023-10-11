import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import gradio as gr  # type: ignore
from fastapi import FastAPI
from llama_index.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.di import root_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chunks.chunks_service import ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings

ingest_service = root_injector.get(IngestService)
chat_service = root_injector.get(ChatService)
chunks_service = root_injector.get(ChunksService)


def _chat(message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
    def yield_deltas(stream: Iterable[ChatResponse | str]) -> Iterable[str]:
        full_response: str = ""
        for delta in stream:
            if isinstance(delta, str):
                full_response += str(delta)
            elif isinstance(delta, ChatResponse):
                full_response += delta.delta or ""
            yield full_response

    def build_history() -> list[ChatMessage]:
        history_messages: list[ChatMessage] = list(
            itertools.chain(
                *[
                    [
                        ChatMessage(content=interaction[0], role=MessageRole.USER),
                        ChatMessage(content=interaction[1], role=MessageRole.ASSISTANT),
                    ]
                    for interaction in history
                ]
            )
        )

        # max 20 messages to try to avoid context overflow
        return history_messages[:20]

    new_message = ChatMessage(content=message, role=MessageRole.USER)
    all_messages = [*build_history(), new_message]
    match mode:
        case "Query Documents":
            query_stream = chat_service.stream_chat(
                messages=all_messages,
                use_context=True,
            )
            yield from yield_deltas(query_stream)

        case "LLM Chat":
            llm_stream = chat_service.stream_chat(
                messages=all_messages,
                use_context=False,
            )
            yield from yield_deltas(llm_stream)

        case "Query Chunks":
            response = chunks_service.retrieve_relevant(
                text=message,
                limit=2,
                prev_next_chunks=1,
            ).__iter__()
            yield "```" + json.dumps(
                [node.__dict__ for node in response],
                default=lambda o: o.__dict__,
                indent=2,
            )


def _list_ingested_files() -> str:
    files = set()
    for ingested_document in ingest_service.list_ingested():
        if ingested_document.doc_metadata is not None:
            files.add(
                ingested_document.doc_metadata.get("file_name") or "[FILE NAME MISSING]"
            )

    return "\n".join(files)


with gr.Blocks() as blocks:
    # Upload button
    file_output = gr.components.Textbox(
        label="Ingested files",
        render=False,
        value=_list_ingested_files,
        every=5,
        lines=5,
        interactive=False,
    )
    upload_button = gr.components.UploadButton(
        "Click to Upload a File",
        type="file",
        file_count="single",
        size="sm",
        render=False,
    )

    def _upload_file(file: TextIO) -> None:
        path = Path(file.name)
        ingest_service.ingest(file_name=path.name, file_data=path)

    upload_button.upload(_upload_file, upload_button)

    # Action dropdown
    dropdown = gr.components.Dropdown(
        choices=["Query Documents", "LLM Chat", "Query Chunks"],
        label="Mode",
        value="Query Documents",
        render=False,
    )

    # Chat
    chat_box = gr.ChatInterface(
        fn=_chat,
        examples=[],
        title="PrivateGPT",
        analytics_enabled=False,
        additional_inputs=[dropdown, upload_button, file_output],
    )


def mount_in_app(app: FastAPI) -> None:
    blocks.queue()
    gr.mount_gradio_app(app, blocks, path=settings.ui.path)
