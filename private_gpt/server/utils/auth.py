"""Authentication mechanism for the API.

When `server.auth.enabled` is True and `user_auth.enabled` is False:
  → uses simple bearer-token comparison (original behaviour).

When `user_auth.enabled` is True:
  → verifies HMAC-signed tokens issued by TokenService.
  → also exposes `get_current_user()` to retrieve the authenticated username.

When both are False:
  → all requests are accepted without a token (original open-access behaviour).
"""

# mypy: ignore-errors
import logging
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from private_gpt.settings.settings import settings

NOT_AUTHENTICATED = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": 'Bearer realm="PrivateGPT", charset="UTF-8"'},
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_bearer(authorization: str) -> str:
    """Strip 'Bearer ' prefix if present."""
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization


# ── authenticated dependency ──────────────────────────────────────────────────

if settings().user_auth.enabled:
    logger.info("User/group auth enabled — using HMAC token authentication")

    def authenticated(
        request: Request,
        authorization: Annotated[str, Header()] = "",
    ) -> bool:
        from private_gpt.server.auth.token_service import TokenService

        token_svc: TokenService = request.state.injector.get(TokenService)
        token = _extract_bearer(authorization)
        if token_svc.verify_token(token) is None:
            raise NOT_AUTHENTICATED
        return True

    def get_current_user(
        request: Request,
        authorization: Annotated[str, Header()] = "",
    ) -> str | None:
        from private_gpt.server.auth.token_service import TokenService

        token_svc: TokenService = request.state.injector.get(TokenService)
        token = _extract_bearer(authorization)
        return token_svc.verify_token(token)

elif settings().server.auth.enabled:
    logger.info("Simple bearer-token authentication enabled")

    def _simple_authentication(
        authorization: Annotated[str, Header()] = "",
    ) -> bool:
        if not secrets.compare_digest(authorization, settings().server.auth.secret):
            raise NOT_AUTHENTICATED
        return True

    def authenticated(
        _simple_authentication: Annotated[bool, Depends(_simple_authentication)],
    ) -> bool:
        return _simple_authentication

    def get_current_user(
        authorization: Annotated[str, Header()] = "",
    ) -> str | None:
        return None  # No user concept in simple token mode

else:
    logger.debug(
        "No authentication configured — all requests are accepted"
    )

    def authenticated() -> bool:
        return True

    def get_current_user() -> str | None:
        return None
