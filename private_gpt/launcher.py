"""FastAPI app creation, logger configuration and main API routes."""
import asyncio
import concurrent
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import nest_asyncio  # type: ignore
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from injector import Injector
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings

from private_gpt.components.persistence.persistence_component import (
    PersistenceComponent,
)
from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import set_global_injector
from private_gpt.docs import DESCRIPTION, TITLE, configure_openapi
from private_gpt.eager_loading import eager_loading
from private_gpt.global_handler import (
    ExceptionMiddleware,
    request_validation_exception_adapter,
)
from private_gpt.initialize import initialize_globals, initialize_observability
from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.chat_async.chat_async_router import (
    chat_router as chat_async_router,
)
from private_gpt.server.completion.completion_router import completion_router
from private_gpt.server.content.content_router import content_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.files.files_router import files_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.convert_router import convert_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.server.models.models_router import models_router
from private_gpt.server.primitives.primitives_router import primitives_router
from private_gpt.server.skills.skill_router import skill_router
from private_gpt.server.tools.tool_router import tool_router
from private_gpt.settings.settings import Settings
from private_gpt.utils.runner import get_version

logger = logging.getLogger(__name__)
UI_DIRECTORY = PROJECT_ROOT_PATH / "ui"


def apply_migrations(injector: Injector) -> None:
    """Ensure that all migrations are applied."""
    logger.debug("Ensuring migrations are applied")
    persistence_component = injector.get(PersistenceComponent)
    persistence_component.apply_migrations()


def create_app(root_injector: Injector) -> FastAPI:
    # Initialize global settings and dependencies
    initialize_globals()
    set_global_injector(root_injector)

    # Retrieve settings and server version
    settings = root_injector.get(Settings)
    version = get_version()

    # Initialize Observability module
    initialize_observability(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager to initialize and clean up resources."""
        # Set nested loop
        nest_asyncio.apply()

        # Set default thread pool limit. This executor now only serves genuine
        # blocking-I/O offloads (broker waits, sync HTTP, sync file reads); all
        # CPU-bound work is routed to dedicated workers, and chat can be routed
        # to a long-lived external worker when ``scheduler.chat.mode`` is enabled,
        # so a small I/O-only pool is enough
        # and stops the GIL from being contended with the event loop.
        cpu_count = os.cpu_count() or 1
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(500, cpu_count * 50), thread_name_prefix="Stream-Pool"
        )
        asyncio.get_running_loop().set_default_executor(executor)

        # Set the global injector as loop injector.
        set_global_injector(root_injector)

        # Ensure migrations are applied
        apply_migrations(root_injector)

        # Eagerly load minimum components
        eager_loading(root_injector)

        # Set up global settings for LlamaIndex
        app.state.injector = root_injector

        # Yield control back to the FastAPI app
        yield

        # Clean up the thread pool executor
        executor.shutdown(wait=True)

        # Clean up resources if necessary
        logger.debug("Cleaning up resources")

    # Start the API
    app = FastAPI(
        # Enable debug mode in FastAPI
        debug=settings.server.debug_mode,
        # Allow to configure prefix for all routes
        root_path=settings.server.root_path,
        root_path_in_servers=True,
        servers=[{"url": settings.server.root_path}]
        if settings.server.root_path
        else None,
        # Use lifespan context manager for initialization
        lifespan=lifespan,
        # Configure Zylon info
        title=TITLE,
        description=DESCRIPTION,
        version=version,
        # Enable OpenAPI schema and Swagger UI
        docs_url=settings.server.api_doc.swagger_url
        if settings.server.api_doc.enabled
        else None,
        redoc_url=settings.server.api_doc.redoc_url
        if settings.server.api_doc.enabled
        else None,
        openapi_url=settings.server.api_doc.openapi_url
        if settings.server.api_doc.enabled
        else None,
    )

    # Disable health logs
    def filter_health_logs(record: Any) -> bool:
        return len(record.args) >= 3 and record.args[2] != "/health"

    logging.getLogger("uvicorn.access").addFilter(filter_health_logs)

    # Add a global exception handler
    app.add_middleware(ExceptionMiddleware)
    app.add_exception_handler(
        RequestValidationError, request_validation_exception_adapter
    )

    # Add a middleware than inject the injector to the request state
    @app.middleware("http")
    async def inject_injector_middleware(request: Request, call_next: Any) -> Any:
        """Middleware to inject the injector into the request state."""
        request.state.injector = (
            request.app.state.injector
            if hasattr(request.app, "state") and hasattr(request.app.state, "injector")
            else root_injector
        )
        response = await call_next(request)
        return response

    app.include_router(chat_router)
    app.include_router(completion_router)
    app.include_router(models_router)
    app.include_router(chat_async_router)
    app.include_router(embeddings_router)
    app.include_router(ingest_router)
    app.include_router(convert_router)
    app.include_router(content_router)
    app.include_router(primitives_router)
    app.include_router(tool_router)
    app.include_router(skill_router)
    app.include_router(files_router)
    app.include_router(health_router)

    if settings.server.ui.enabled:
        if not UI_DIRECTORY.exists():
            raise RuntimeError(
                f"UI hosting enabled but UI directory does not exist: {UI_DIRECTORY}"
            )

        logger.debug(f"UI enabled at {settings.server.ui.path}")
        ui_path = settings.server.ui.path.rstrip("/") or "/"

        if ui_path != "/":

            @app.get(ui_path, include_in_schema=False)
            async def redirect_ui_index(request: Request) -> RedirectResponse:
                root_path = request.scope.get("root_path", "").rstrip("/")
                return RedirectResponse(url=f"{root_path}{ui_path}/", status_code=307)

        connect_host = (
            "127.0.0.1" if settings.server.host == "0.0.0.0" else settings.server.host
        )
        server_url = f"http://{connect_host}:{settings.server.port}"
        if settings.server.root_path:
            server_url += "/" + settings.server.root_path.strip("/")
        _index_html = (UI_DIRECTORY / "index.html").read_text()
        _index_html = _index_html.replace(
            'const DEFAULT_BASE_URL = window.location.origin === "null" ? "http://127.0.0.1:8080" : window.location.origin;',
            f'const DEFAULT_BASE_URL = "{server_url}";',
        )

        @app.get(f"{ui_path}/", include_in_schema=False)
        @app.get(f"{ui_path}/index.html", include_in_schema=False)
        async def serve_ui_index() -> HTMLResponse:
            return HTMLResponse(content=_index_html)

        static_files = StaticFiles(directory=UI_DIRECTORY, html=True)

        async def _ui_app(scope: Any, receive: Any, send: Any) -> None:
            # When a reverse proxy strips the root_path prefix before forwarding
            # (prefix rewriting), scope["path"] still contains the mount prefix
            # (e.g. "/ui/") while scope["root_path"] becomes "/gpt/ui". Starlette's
            # StaticFiles.get_route_path() then fails to strip anything and tries to
            # serve a non-existent "ui" subdirectory. Normalize the path here so
            # StaticFiles works correctly regardless of proxy rewriting.
            path: str = scope.get("path", "")
            if ui_path != "/" and (path == ui_path or path.startswith(ui_path + "/")):
                remaining = path[len(ui_path) :] or "/"
                scope = dict(scope)
                scope["path"] = remaining
            await static_files(scope, receive, send)

        app.mount(ui_path or "/", _ui_app, name="ui")

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

    # Set global embedding model to Mock to prevent LlamaIndex to default to use OpenAI
    LlamaIndexSettings.embed_model = MockEmbedding(384)

    # Configure OpenAPI schema
    configure_openapi(app)

    return app
