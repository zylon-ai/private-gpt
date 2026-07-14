from private_gpt.arq.hooks import (
    on_job_end,  # noqa: F401 — used by runner via worker_module.on_job_end
)
from private_gpt.arq.lifecycle import (  # noqa: F401 — used by runner via worker_module.*
    shutdown,
    startup,
)
from private_gpt.arq.tasks import autodiscover_tasks

functions = autodiscover_tasks()
