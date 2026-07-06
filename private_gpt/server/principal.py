from __future__ import annotations

from contextvars import ContextVar


class Principal:
    """Request-scoped principal carrying HTTP headers and cookies.

    Set once at the HTTP middleware layer and accessible from anywhere in the
    call tree via ``Principal.current()`` — no parameter threading needed.

    Usage::

        auth = Principal.current().authorization
        key = Principal.current().api_key
        session = Principal.current().cookies.get("session")
        custom = Principal.current().headers.get("x-custom-header")
    """

    __slots__ = ("_cookies", "_headers")

    _context: ContextVar[Principal | None] = ContextVar("principal", default=None)

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        self._headers: dict[str, str] = dict(headers) if headers else {}
        self._cookies: dict[str, str] = dict(cookies) if cookies else {}

    # -- Typed claims ----------------------------------------------------------

    @property
    def headers(self) -> dict[str, str]:
        """All captured HTTP request headers (lowercased names)."""
        return self._headers

    @property
    def cookies(self) -> dict[str, str]:
        """All captured HTTP request cookies."""
        return self._cookies

    @property
    def authorization(self) -> str | None:
        """Full ``Authorization`` header value (e.g. ``Bearer sk-abc123``)."""
        return self._headers.get("authorization")

    @property
    def api_key(self) -> str | None:
        """Bearer token extracted from the Authorization header.

        Returns the raw key without the ``Bearer `` prefix.
        """
        auth = self.authorization
        if auth and auth.lower().startswith("bearer "):
            return auth[7:]
        return auth

    @property
    def api_key_header(self) -> str | None:
        """Raw ``x-api-key`` header value (used by Anthropic SDK for API key auth)."""
        return self._headers.get("x-api-key")

    # -- Context management ----------------------------------------------------

    @classmethod
    def current(cls) -> Principal:
        """Return the current request-scoped principal.

        Never returns ``None`` — an anonymous principal with no claims is
        returned when no principal has been set.
        """
        p = cls._context.get()
        return p if p is not None else cls()

    @classmethod
    def reset(cls) -> None:
        """Remove the current principal from the async context."""
        cls._context.set(None)

    def set_current(self) -> Principal:
        """Store this principal as the current one for the async context."""
        self._context.set(self)
        return self

    # -- Helpers ---------------------------------------------------------------

    @property
    def anonymous(self) -> bool:
        """``True`` when no claims have been populated."""
        return not self._headers and not self._cookies

    def as_env(self, *, prefix: str = "ANTHROPIC") -> dict[str, str]:
        """Return principal credentials as env vars for subprocess tools.

        * ``x-api-key`` header → ``{PREFIX}_API_KEY``
        * ``Authorization`` header → ``{PREFIX}_AUTH_TOKEN``
        """
        result: dict[str, str] = {}
        if self.api_key_header:
            result[f"{prefix}_API_KEY"] = self.api_key_header
        if self.authorization:
            result[f"{prefix}_AUTH_TOKEN"] = self.authorization

        return result

    def resolve_env(self, env_vars: dict[str, str]) -> dict[str, str]:
        """Resolve ``$PRINCIPAL_*`` sentinels in *env_vars* against this principal.

        Supported sentinels:

        * ``$PRINCIPAL_AUTH_TOKEN`` → ``authorization`` header (e.g. ``Bearer sk-...``)
        * ``$PRINCIPAL_BEARER`` → ``api_key`` (Bearer token without the prefix)
        * ``$PRINCIPAL_API_KEY`` → ``x-api-key`` header (API key sent as ``X-Api-Key``)

        ``$PRINCIPAL_*`` sentinels with no resolved value are omitted.
        """
        sentinel_map: dict[str, str | None] = {
            "$PRINCIPAL_AUTH_TOKEN": self.authorization,
            "$PRINCIPAL_BEARER": self.api_key,
            "$PRINCIPAL_API_KEY": self.api_key_header,
        }
        result: dict[str, str] = {}
        for key, value in env_vars.items():
            if value in sentinel_map:
                resolved = sentinel_map[value]
                if resolved:
                    result[key] = resolved
            elif value and not value.startswith("$PRINCIPAL_"):
                result[key] = value
        return result

    def __repr__(self) -> str:
        parts: list[str] = []
        if "authorization" in self._headers:
            parts.append("authorization=***")
        other_headers = [k for k in self._headers if k != "authorization"]
        if other_headers:
            parts.append(f"headers=[{', '.join(other_headers)}]")
        if self._cookies:
            parts.append(f"cookies=[{', '.join(self._cookies.keys())}]")
        return f"Principal({', '.join(parts)})" if parts else "Principal(anonymous)"
