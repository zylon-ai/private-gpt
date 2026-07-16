import os
from typing import Any

import nest_asyncio

from private_gpt.di import (
    clean_global_injector,
    get_global_injector,
    set_global_injector,
)
from private_gpt.eager_loading import warm
from private_gpt.initialize import initialize_globals, initialize_observability
from private_gpt.settings.settings import settings


async def startup(ctx: dict[Any, Any]) -> None:
    current_settings = settings()
    initialize_globals()
    initialize_observability(current_settings)
    nest_asyncio.apply()
    injector = get_global_injector(allow_to_generate_new_injectors=True)
    set_global_injector(injector)
    warm_profile = os.environ.get("PGPT_WORKER_WARM_PROFILE", "").strip()
    if warm_profile:
        warm(injector, profile=warm_profile)
    ctx["injector"] = injector


async def shutdown(ctx: dict[Any, Any]) -> None:
    del ctx
    await clean_global_injector()
