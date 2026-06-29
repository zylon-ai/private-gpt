from __future__ import annotations

from contextvars import ContextVar


class Principal:
    """Request-scoped principal with known, typed claims.

    Set once at the HTTP middleware layer and accessible from anywhere in the
    call tree via ``Principal.current()`` — no parameter threading needed.

    Usage::

        api_key = Principal.current().api_key
        features = Principal.current().anthropic_beta
    """

    __slots__ = (
        "_anthropic_beta",
        "_api_key",
        "_authorization",
    )

    _context: ContextVar[Principal | None] = ContextVar("principal", default=None)

    def __init__(
        self,
        *,
        authorization: str | None = None,
        api_key: str | None = None,
        anthropic_beta: list[str] | None = None,
    ) -> None:
        self._authorization = authorization
        self._api_key = api_key
        self._anthropic_beta = anthropic_beta

    # -- Known claims ----------------------------------------------------------

    @property
    def authorization(self) -> str | None:
        """Full ``Authorization`` header value (e.g. ``Bearer sk-abc123``)."""
        return self._authorization

    @property
    def api_key(self) -> str | None:
        """Bearer token extracted from the Authorization header.

        Returns the raw key without the ``Bearer `` prefix.  If an explicit
        ``api_key`` was provided at construction time it takes precedence.
        """
        if self._api_key is not None:
            return self._api_key
        auth = self._authorization
        if auth and auth.startswith("Bearer "):
            return auth[7:]
        return auth

    @property
    def anthropic_beta(self) -> list[str] | None:
        """``anthropic-beta`` header feature flags (e.g. prompt-caching)."""
        return self._anthropic_beta

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
        return (
            self._authorization is None
            and self._api_key is None
            and self._anthropic_beta is None
        )

    def as_env(self, *, prefix: str = "ANTHROPIC") -> dict[str, str]:
        """Return claims as ``{PREFIX_API_KEY: ..., ...}`` env vars."""
        env: dict[str, str] = {}
        key = self.api_key
        if key:
            env[f"{prefix}_API_KEY"] = key
        beta = self.anthropic_beta
        if beta:
            env[f"{prefix}_BETA"] = ",".join(beta)
        return env

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._authorization:
            parts.append("authorization=***")
        if self._api_key:
            parts.append("api_key=***")
        if self._anthropic_beta:
            parts.append(f"anthropic_beta={self._anthropic_beta}")
        return f"Principal({', '.join(parts)})" if parts else "Principal(anonymous)"
