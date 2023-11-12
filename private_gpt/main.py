"""FastAPI app creation, logger configuration and main API routes."""

import llama_index

from private_gpt.di import global_injector
from private_gpt.launcher import create_app

# Add LlamaIndex simple observability
llama_index.set_global_handler("simple")

app = create_app(global_injector)
