"""HMAC-signed session token creation and verification.

Tokens are self-contained, stateless, and do not require a database lookup
on every request — only signature and expiry validation.

Token format (URL-safe base64, no padding):
    <payload_b64>.<signature_b64>

where payload is:  "<username>:<unix_timestamp_int>"
"""

import base64
import hmac
import logging
import time
from hashlib import sha256

from injector import inject, singleton

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_ENCODING = "utf-8"


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode(_ENCODING)


def _b64_decode(s: str) -> bytes:
    # Re-add padding
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (padding % 4))


@singleton
class TokenService:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._secret = settings.server.auth.secret.encode(_ENCODING)
        self._expiry_seconds = settings.user_auth.token_expiry_hours * 3600

    def _sign(self, payload: str) -> str:
        return _b64_encode(
            hmac.new(self._secret, payload.encode(_ENCODING), sha256).digest()
        )

    def create_token(self, username: str) -> str:
        """Return a signed token for *username*."""
        payload = f"{username}:{int(time.time())}"
        payload_b64 = _b64_encode(payload.encode(_ENCODING))
        sig = self._sign(payload_b64)
        return f"{payload_b64}.{sig}"

    def verify_token(self, token: str) -> str | None:
        """Verify *token* and return the username, or None if invalid/expired."""
        try:
            payload_b64, sig = token.split(".", 1)
        except ValueError:
            return None

        # Constant-time signature check
        expected_sig = self._sign(payload_b64)
        if not hmac.compare_digest(sig, expected_sig):
            return None

        try:
            payload = _b64_decode(payload_b64).decode(_ENCODING)
            username, ts_str = payload.rsplit(":", 1)
            ts = int(ts_str)
        except (ValueError, UnicodeDecodeError):
            return None

        if time.time() - ts > self._expiry_seconds:
            logger.debug("Token expired for user=%s", username)
            return None

        return username
