"""This file should be imported only and only if you want to run the UI locally."""
import itertools
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
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

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "\n\n Sources: \n"

MODES = ["Query Docs", "Search in Docs", "LLM Chat"]

from private_gpt.ui.common import PrivateGpt

@singleton
class UsersUI(PrivateGpt):

    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
    ) -> None:
        super().__init__(ingest_service, chat_service, chunks_service)

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
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }",
        ) as users:
            # with gr.Row():
            #     gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    mode = gr.Radio(
                        MODES,
                        label="Mode",
                        value="Query Docs",
                    )
                    ingested_dataset = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        interactive=False,
                        render=False,  # Rendered under the button
                    )
                    ingested_dataset.change(
                        self._list_ingested_files,
                        outputs=ingested_dataset,
                    )
                    ingested_dataset.render()
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    # When mode changes, set default system prompt
                    mode.change(
                        self._set_current_mode, inputs=mode, outputs=system_prompt_input
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                with gr.Column(scale=7, elem_id="col"):
                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=f"LLM: {settings().llm.mode}",
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            # avatar_images=(
                            #     None,
                            #     AVATAR_BOT,
                            # ),
                        ),
                        additional_inputs=[mode, system_prompt_input],
                    )
        return users

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        logger.info("PATH---------------------------->:%s", path)
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the regular gradio UI at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path)
    
if __name__ == "__main__":
    ui = global_injector.get(UsersUI)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
        
