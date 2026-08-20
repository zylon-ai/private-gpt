from __future__ import annotations

from private_gpt.context import current_bag, reset_bag

_HEADERS_KEY = "headers"
_COOKIES_KEY = "cookies"


class Principal:
    """Request-scoped principal carrying HTTP headers and cookies.

    The principal is a typed view over the shared context bag (see
    ``private_gpt.context``). ``Principal.current()`` reads the bag and exposes
    the request's headers/cookies as typed claims — no parameter threading
    needed. Because the bag travels with ARQ / Celery jobs, the principal is
    automatically available on workers.

    A ``Principal`` constructed with headers/cookies holds *pending claims* that
    are installed into the bag by ``set_current()`` (used by the HTTP
    middleware).

    Usage::

        auth = Principal.current().authorization
        key = Principal.current().api_key
        session = Principal.current().cookies.get("session")
        custom = Principal.current().headers.get("x-custom-header")
    """

    __slots__ = ("_cookies", "_headers")

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        # Pending claims, installed into the bag by ``set_current()``.
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
        return self.headers.get("authorization")

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
        return self.headers.get("x-api-key")

    # -- Context management ----------------------------------------------------

    @classmethod
    def current(cls) -> Principal:
        """Return the current request-scoped principal.

        Backed by the context bag, so on ARQ/Celery workers the transported
        bag provides the claims. Never returns ``None`` — an anonymous
        principal with no claims is returned when no principal has been set.
        """
        bag = current_bag()
        return cls(
            headers=bag.get(_HEADERS_KEY),
            cookies=bag.get(_COOKIES_KEY),
        )

    @classmethod
    def reset(cls) -> None:
        """Clear the context bag entirely for the current request.

        The whole bag is dropped (not just the principal's keys) so no stale
        request context leaks into the next request on the same asyncio task.
        """
        reset_bag()

    def set_current(self) -> Principal:
        """Store this principal's claims into the context bag.

        Copy-on-write: replaces the bag rather than mutating it in place, so
        ``asyncio.create_task`` contexts (which snapshot the ContextVar by
        reference) do not observe each other's mutations.
        """
        from private_gpt.context import replace_bag

        replace_bag({_HEADERS_KEY: self._headers, _COOKIES_KEY: self._cookies})
        return self

    # -- Helpers ---------------------------------------------------------------

    @property
    def anonymous(self) -> bool:
        """``True`` when no claims have been populated."""
        return not self.headers and not self.cookies

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
        if "authorization" in self.headers:
            parts.append("authorization=***")
        other_headers = [k for k in self.headers if k != "authorization"]
        if other_headers:
            parts.append(f"headers=[{', '.join(other_headers)}]")
        if self.cookies:
            parts.append(f"cookies=[{', '.join(self.cookies.keys())}]")
        return f"Principal({', '.join(parts)})" if parts else "Principal(anonymous)"
