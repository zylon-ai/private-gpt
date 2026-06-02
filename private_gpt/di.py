import asyncio
import logging
import threading
from asyncio import AbstractEventLoop
from typing import Any, cast

from injector import Injector

from private_gpt.settings.settings import Settings, unsafe_typed_settings

_INJECTOR_KEY = "_injector"
_ALLOWED_CREATION_NEW_INJECTORS = True

_global_injector_lock = threading.RLock()
_global_injector: Injector | None = None

_loop_injector_lock = threading.RLock()


class InjectorNotFoundError(Exception):
    pass


logger = logging.getLogger(__name__)
logger.setLevel(
    logging.DEBUG if unsafe_typed_settings.server.debug_mode else logging.INFO
)


def create_application_injector() -> Injector:
    """Create a new injector with the default bindings."""
    _injector = Injector(auto_bind=True)
    _injector.binder.bind(Settings, to=unsafe_typed_settings)
    return _injector


def get_injector(
    allow_to_generate_new_injectors: bool = _ALLOWED_CREATION_NEW_INJECTORS,
) -> Injector:
    """Get the injector from the current asyncio loop or global fallback.

    First tries to get the injector from the current asyncio loop.
    If not running in an asyncio loop or no injector is set,
    falls back to the global injector.
    """
    global _global_injector
    try:
        loop = asyncio.get_running_loop()

        injector = getattr(loop, _INJECTOR_KEY, None)
        if injector is not None:
            return cast(Injector, injector)

        with _loop_injector_lock:
            if _global_injector is not None:
                injector = _global_injector
            else:
                logging.debug(
                    "No injector found in the current asyncio loop. "
                    "Creating a new one and setting it in the loop.",
                )
                injector = create_application_injector()
            tmp_injector = getattr(loop, _INJECTOR_KEY, None)
            if tmp_injector is None:
                setattr(loop, _INJECTOR_KEY, injector)
        if not allow_to_generate_new_injectors:
            raise InjectorNotFoundError(
                "No injector set in the current asyncio loop. "
                "PLEASE REVIEW YOUR USAGE OF THIS FUNCTION!",
            )
        return get_injector()
    except RuntimeError:
        # Not in an asyncio loop, use global injector
        if _global_injector is None:
            with _global_injector_lock:
                global_injector = create_application_injector()
                if _global_injector is None:
                    _global_injector = global_injector
        return _global_injector


def set_injector(injector: Injector) -> None:
    """Set the injector in the current asyncio loop or globally.

    If running in an asyncio loop, stores the injector in the loop.
    Otherwise, sets it as the global injector.
    """
    try:
        with _loop_injector_lock:
            loop = asyncio.get_running_loop()
            setattr(loop, _INJECTOR_KEY, injector)
    except RuntimeError:
        # Not in an asyncio loop, set global injector
        with _global_injector_lock:
            global _global_injector
            _global_injector = injector


def clean_global_injector(loop: AbstractEventLoop | None = None) -> None:
    try:
        loop = loop or asyncio.get_running_loop()
        if hasattr(loop, _INJECTOR_KEY):
            logger.debug("Closing loop injector resources...")
            _injector = cast(Injector, getattr(loop, _INJECTOR_KEY))
            for interface, _ in _injector.binder._bindings.items():
                impl: Any = _injector.get(interface)
                if hasattr(impl, "close"):
                    impl.close()

                del impl

            with _loop_injector_lock:
                delattr(loop, _INJECTOR_KEY)
    except RuntimeError:
        # Not in an asyncio loop, do nothing
        pass


def get_global_injector(
    allow_to_generate_new_injectors: bool = _ALLOWED_CREATION_NEW_INJECTORS,
) -> Injector:
    return get_injector(allow_to_generate_new_injectors)


def set_global_injector(injector: Injector) -> None:
    set_injector(injector)
