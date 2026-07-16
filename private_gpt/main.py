"""FastAPI app creation, logger configuration and main API routes."""

from private_gpt.di import get_global_injector
from private_gpt.launcher import create_app

app = create_app(get_global_injector())
