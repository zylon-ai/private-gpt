"""FastAPI app creation, logger configuration and main API routes."""
import logging
from typing import Any

import llama_index
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from private_gpt.paths import docs_path
from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

# Add LlamaIndex simple observability
llama_index.set_global_handler("simple")

# Start the API
with open(docs_path / "description.md") as description_file:
    description = description_file.read()

tags_metadata = [
    {
        "name": "Ingestion",
        "description": "High-level APIs covering document ingestion -internally "
        "managing document parsing, splitting,"
        "metadata extraction, embedding generation and storage- and ingested "
        "documents CRUD."
        "Each ingested document is identified by an ID that can be used to filter the "
        "context"
        "used in *Contextual Completions* and *Context Chunks* APIs.",
    },
    {
        "name": "Contextual Completions",
        "description": "High-level APIs covering contextual Chat and Completions. They "
        "follow OpenAI's format, extending it to "
        "allow using the context coming from ingested documents to create the "
        "response. Internally"
        "manage context retrieval, prompt engineering and the response generation.",
    },
    {
        "name": "Context Chunks",
        "description": "Low-level API that given a query return relevant chunks of "
        "text coming from the ingested"
        "documents.",
    },
    {
        "name": "Embeddings",
        "description": "Low-level API to obtain the vector representation of a given "
        "text, using an Embeddings model."
        "Follows OpenAI's embeddings API format.",
    },
    {
        "name": "Health",
        "description": "Simple health API to make sure the server is up and running.",
    },
]

app = FastAPI()


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="PrivateGPT",
        description=description,
        version="0.1.0",
        summary="PrivateGPT is a production-ready AI project that allows you to "
        "ask questions to your documents using the power of Large Language "
        "Models (LLMs), even in scenarios without Internet connection. "
        "100% private, no data leaves your execution environment at any point.",
        contact={
            "url": "https://github.com/imartinez/privateGPT",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        routes=app.routes,
        tags=tags_metadata,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://lh3.googleusercontent.com/drive-viewer"
        "/AK7aPaD_iNlMoTquOBsw4boh4tIYxyEuhz6EtEs8nzq3yNkNAK00xGj"
        "E1KUCmPJSk3TYOjcs6tReG6w_cLu1S7L_gPgT9z52iw=s2560"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]

app.include_router(completions_router)
app.include_router(chat_router)
app.include_router(chunks_router)
app.include_router(ingest_router)
app.include_router(embeddings_router)
app.include_router(health_router)

if settings.server.cors.enabled:
    logger.debug("Setting up CORS middleware")
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=settings.server.cors.allow_credentials,
        allow_origins=settings.server.cors.allow_origins,
        allow_origin_regex=settings.server.cors.allow_origin_regex,
        allow_methods=settings.server.cors.allow_methods,
        allow_headers=settings.server.cors.allow_headers,
    )


if settings.ui.enabled:
    logger.debug("Importing the UI module")
    from private_gpt.ui.ui import PrivateGptUi

    PrivateGptUi().mount_in_app(app)
