import contextlib
from typing import Any

from private_gpt.di import get_global_injector, set_global_injector


async def on_job_end(ctx: dict[Any, Any]) -> None:
    del ctx
    with contextlib.suppress(Exception):
        injector = get_global_injector(allow_to_generate_new_injectors=True)
        set_global_injector(injector)
