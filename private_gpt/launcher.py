"""FastAPI app creation, logger configuration and main API routes."""

import logging

from injector import Injector
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.settings.settings import Settings
from private_gpt.users.api.v1.api import api_router
from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.components.ocr_components.table_ocr_api import pdf_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router

logger = logging.getLogger(__name__)


def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(dependencies=[Depends(bind_injector_to_request)])
    app.include_router(completions_router)
    app.include_router(chat_router)
    app.include_router(chunks_router)
    app.include_router(ingest_router)
    app.include_router(embeddings_router)
    app.include_router(health_router)
    app.include_router(api_router)
    app.include_router(pdf_router)
    
    # Add LlamaIndex simple observability
    global_handler = create_global_handler("simple")
    LlamaIndexSettings.callback_manager = CallbackManager([global_handler])

    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=["http://localhost:3000/", "http://10.1.101.125:80", "http://quickgpt.gibl.com.np:80", "http://127.0.0.1",
                           "http://10.1.101.125", "http://quickgpt.gibl.com.np", "http://localhost:8000", "http://192.168.1.93", "http://192.168.1.93:88", 
                           "http://192.168.1.98", "http://192.168.1.98:5173", "http://localhost:3000","https://globaldocquery.gibl.com.np/", "http://127.0.0.1/", "http://localhost/", 
                           "http://localhost:80", "http://192.168.1.131", 'http://192.168.1.131:3000'
                           , "http://192.168.1.127", 'http://192.168.1.127:3000'
                           , "http://192.168.1.70", 'http://192.168.1.70:3000'
                           ],
            allow_methods=["DELETE", "GET", "POST", "PUT", "OPTIONS", "PATCH"],
            allow_headers=["*"],
        )

    return app