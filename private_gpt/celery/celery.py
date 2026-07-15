import os
from collections.abc import Callable, Sequence
from typing import Any

from celery import Celery

from private_gpt.celery.bootsteps import LivenessProbe
from private_gpt.celery.config import CeleryConfig, backend_config, broker_config
from private_gpt.celery.task_registry import get_task_packages
from private_gpt.initialize import initialize_globals, initialize_observability
from private_gpt.settings.settings import Settings, settings


def _configured_task_packages(task_packages: Sequence[str]) -> tuple[str, ...]:
    if task_packages:
        return tuple(task_packages)
    configured = os.environ.get("PGPT_CELERY_TASK_PACKAGES", "")
    return tuple(
        package.strip() for package in configured.split(",") if package.strip()
    )


def _autodiscover_tasks(
    celery_app: Celery, task_packages: Sequence[str], *, force: bool = False
) -> None:
    celery_app.autodiscover_tasks(list(task_packages), force=force)


def _enable_sync_send_task(
    celery_app: Celery,
    current_settings: Settings,
    task_packages: Sequence[str],
) -> None:
    if current_settings.celery.use_workers:
        return

    def send_task_sync(name, args=(), kwargs=None, **opts) -> Any:  # type: ignore
        # Execute the task immediately, instead of using a worker
        # Same as always_eager mode, but for send_task()
        if kwargs is None:
            kwargs = {}

        # Force task discovery if not already done
        if name not in celery_app.tasks:
            _autodiscover_tasks(celery_app, task_packages, force=True)

        # Get the task and execute it
        task = celery_app.tasks[name]
        return task.apply(args, kwargs, **opts)

    # Monkey-patching of Celery send_task to make it run synchronously
    celery_app.send_task = send_task_sync  # type: ignore


def create_app(
    *,
    settings_resolver: Callable[[], Settings] = settings,
    before_setup: Callable[[], None] | None = None,
    task_packages: Sequence[str] = (),
    app_name: str = __name__,
) -> Celery:
    if before_setup is not None:
        before_setup()

    resolved_task_packages = get_task_packages(
        *_configured_task_packages(task_packages)
    )

    celery_app = Celery(
        app_name,
        broker=broker_config.url,
        backend=backend_config.url,
    )

    # Configure Celery settings
    celery_app.config_from_object(CeleryConfig)

    # Add liveness probe
    celery_app.steps["worker"].add(LivenessProbe)

    current_settings = settings_resolver()

    # Initialize global settings and dependencies
    initialize_globals()

    # Initialize Observability module
    initialize_observability(current_settings)

    # Autodiscover tasks
    _autodiscover_tasks(celery_app, resolved_task_packages)
    _enable_sync_send_task(celery_app, current_settings, resolved_task_packages)

    return celery_app


celery_app: Celery = create_app()
