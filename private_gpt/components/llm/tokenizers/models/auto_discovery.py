import asyncio
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

from private_gpt.components.llm.tokenizers.models.model_discovery import discover_model

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def auto_discover_model(
    enabled: bool = True,
    tokenizer_only: bool = False,
    raise_on_error: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to automatically discover and resolve model identifiers to local paths.

    Intercepts function calls with ``model_id`` and ``cache_dir`` kwargs and resolves
    them to local paths before invoking the wrapped function. Resolution order:
      1. HuggingFace hub cache
      2. Download from HuggingFace if not offline
      3. Falls back to original ``model_id``

    Args:
        enabled: Enable or disable the auto-discovery behavior.
        tokenizer_only: Only resolve tokenizer files.
        raise_on_error: Raise exceptions instead of falling back to original model_id.

    Example:
        @auto_discover_model()
        def load_model(model_id: str, cache_dir: Path, **kwargs):
            return AutoModel.from_pretrained(model_id)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if not enabled:
            return func

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            call_kwargs: dict[str, Any] = dict(kwargs)
            model_id = call_kwargs.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                return func(*args, **kwargs)

            cache_dir_raw = call_kwargs.get("cache_dir")
            if not cache_dir_raw:
                logger.warning("No cache_dir specified, skipping model resolution")
                return func(*args, **kwargs)

            cache_dir = Path(str(cache_dir_raw))
            cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                resolved_id, is_local = await discover_model(
                    model_id=model_id,
                    cache_dir=cache_dir,
                    force_download=bool(call_kwargs.get("force_download", False)),
                    local_files_only=bool(call_kwargs.get("local_files_only", False)),
                    tokenizer_only=tokenizer_only,
                )
                call_kwargs["model_id"] = resolved_id
                if is_local:
                    call_kwargs["local_files_only"] = True

            except Exception as e:
                if raise_on_error:
                    raise
                call_kwargs["local_files_only"] = True
                logger.warning(f"Falling back to original model_id: {model_id}, e={e}")

            if asyncio.iscoroutinefunction(func):
                return cast(T, await func(*args, **call_kwargs))
            return func(*args, **call_kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return asyncio.run(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, T], async_wrapper)
        return cast(Callable[P, T], sync_wrapper)

    return decorator
