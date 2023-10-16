"""FastAPI app creation, logger configuration and main API routes."""
import sys
from typing import Any

import llama_index
from fastapi import FastAPI
from loguru import logger

from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.settings.settings import settings

# Remove pre-configured logging handler
logger.remove(0)
# Create a new logging handler same as the pre-configured one but with the extra
# attribute `request_id`
logger.add(
    sys.stdout,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "ID: {extra[request_id]} - <level>{message}</level>"
    ),
)

# Add LlamaIndex simple observability
llama_index.set_global_handler("simple")

# Start the API
description = """
PrivateGPT API helps you do awesome stuff. 🚀

## Section 1

Add here intro **section 1**.

## Section 2

Add here intro **section 2**.

* **List item 1** lorem ipsum.
* **List item 2** lorem ipsum.
"""

app = FastAPI(
    title="PrivateGPT",
    description=description,
    version="0.1.0",
    summary="PrivateGPT Summary",
    contact={
        "url": "https://github.com/imartinez/privateGPT",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)


@app.get("/health")
def health() -> Any:
    return {"status": "ok"}


app.include_router(completions_router)
app.include_router(chat_router)
app.include_router(chunks_router)
app.include_router(ingest_router)
app.include_router(embeddings_router)


if settings.ui.enabled:
    from private_gpt.ui.ui import mount_in_app

    mount_in_app(app)
