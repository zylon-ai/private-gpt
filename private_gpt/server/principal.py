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
        """Return the API key as a ``{PREFIX_API_KEY: ...}`` env var."""
        key = self.api_key
        if key:
            return {f"{prefix}_API_KEY": key}
        return {}

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
