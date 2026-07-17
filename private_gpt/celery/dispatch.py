from typing import Any


def dispatch_task(
    *,
    task_name: str,
    queue: str,
    args: tuple[Any, ...] | list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    task_id: str | None = None,
    ignore_result: bool | None = None,
) -> Any:
    from private_gpt.celery.celery import celery_app

    options: dict[str, Any] = {"queue": queue}
    if task_id is not None:
        options["task_id"] = task_id
    if ignore_result is not None:
        options["ignore_result"] = ignore_result

    return celery_app.send_task(
        task_name,
        args=args,
        kwargs=kwargs,
        **options,
    )
