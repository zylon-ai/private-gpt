"""PrivateGPT Chainlit UI wrapper.

Provides the ``PrivateGptUi`` singleton that mounts the Chainlit application
inside the existing FastAPI app at the configured path.

The actual Chainlit handlers live in ``chainlit_app.py`` (same package).
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from injector import inject, singleton

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

_THIS_DIRECTORY = Path(__file__).parent


@singleton
class PrivateGptUi:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        try:
            from chainlit.utils import mount_chainlit
        except ImportError as exc:
            raise ImportError(
                "Chainlit is not installed. "
                "Install it with: poetry install --extras ui"
            ) from exc

        chainlit_app_path = str(_THIS_DIRECTORY / "chainlit_app.py")
        mount_chainlit(app=app, target=chainlit_app_path, path=path)
        logger.info("Chainlit UI mounted at path=%s", path)
