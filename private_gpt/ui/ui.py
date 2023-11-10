"""This file should be imported only and only if you want to run the UI locally."""
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from llama_index.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.di import root_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chunks.chunks_service import ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)


UI_TAB_TITLE = "My Private GPT"


class PrivateGptUi:
    def __init__(self) -> None:
        self._ingest_service = root_injector.get(IngestService)
        self._chat_service = root_injector.get(ChatService)
        self._chunks_service = root_injector.get(ChunksService)

        # Cache the UI blocks
        self._ui_block = None

    def _chat(self, message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
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
                            ChatMessage(
                                content=interaction[1], role=MessageRole.ASSISTANT
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

                yield "\n\n\n".join(
                    f"{index}. **{chunk.document.doc_metadata['file_name'] if chunk.document.doc_metadata else ''} "
                    f"(page {chunk.document.doc_metadata['page_label'] if chunk.document.doc_metadata else ''})**\n "
                    f"{chunk.text}"
                    for index, chunk in enumerate(response, start=1)
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

    def _upload_file(self, file: TextIO) -> None:
        path = Path(file.name)
        self._ingest_service.ingest(file_name=path.name, file_data=path)

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: #C7BAFF;"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 25% }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            with gr.Row():
                with gr.Column(scale=3, variant="compact"):
                    mode = gr.Radio(
                        ["Query Docs", "Search in Docs", "LLM Chat"],
                        label="Mode",
                        value="Query Docs",
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload a File",
                        type="file",
                        file_count="single",
                        size="sm",
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                with gr.Column(scale=7):
                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=f"LLM: {settings.llm.mode}",
                            show_copy_button=True,
                            render=False,
                            avatar_images=(
                                None,
                                "https://lh3.googleusercontent.com/drive-viewer/AK7aPa"
                                "AicXck0k68nsscyfKrb18o9ak3BSaWM_Qzm338cKoQlw72Bp0UKN84"
                                "IFZjXjZApY01mtnUXDeL4qzwhkALoe_53AhwCg=s2560",
                            ),
                        ),
                        additional_inputs=[mode, upload_button],
                    )
        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        base_path = settings.ui.path
        logger.info("Mounting the gradio UI, at path=%s", base_path)
        gr.mount_gradio_app(app, blocks, path=base_path)


if __name__ == "__main__":
    ui = PrivateGptUi()
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
