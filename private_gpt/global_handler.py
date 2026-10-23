import logging
from typing import Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from private_gpt.events.event_errors import Errors
from private_gpt.events.models import FatalError

logger = logging.getLogger(__name__)


def _fatal_error_payload(error: BaseException) -> tuple[int, dict[str, Any]]:
    wrapped_exception = Errors.build(error)
    payload = jsonable_encoder(FatalError.from_exception(error).model_dump())
    explanation = payload.get("error", {}).get("detail", {}).get("explanation")
    if explanation is not None:
        payload["detail"] = explanation
    return wrapped_exception.status_code, payload


class ExceptionMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Any) -> None:
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logger.error("Unhandled exception occurred", exc_info=exc)
            response = self._create_error_response(exc)
            await response(scope, receive, send)

    def _create_error_response(self, exc: Exception) -> JSONResponse:
        """Create appropriate error response based on exception type."""
        status_code, fatal_error = _fatal_error_payload(exc)
        return JSONResponse(
            status_code=status_code,
            content=jsonable_encoder(fatal_error),
        )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    del request  # Request currently unused.
    wrapped_exception = Errors.InvalidRequest(
        "Request validation failed",
        event_code=Errors.Codes.INVALID_REQUEST_ERROR,
        status_code=400,
        original_exception=exc,
    )
    _, payload = _fatal_error_payload(wrapped_exception)
    return JSONResponse(status_code=400, content=jsonable_encoder(payload))


async def request_validation_exception_adapter(
    request: Request, exc: Exception
) -> JSONResponse:
    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)

    status_code, payload = _fatal_error_payload(exc)
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(payload),
    )
