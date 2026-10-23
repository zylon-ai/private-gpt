BASE_TASK_PACKAGES = (
    "private_gpt.celery.tasks.ingestion",
    "private_gpt.celery.tasks.tools",
)

_EXTERNAL_TASK_PACKAGES: list[str] = []


def register_task_packages(*task_packages: str) -> None:
    for task_package in task_packages:
        if task_package not in _EXTERNAL_TASK_PACKAGES:
            _EXTERNAL_TASK_PACKAGES.append(task_package)


def get_task_packages(*task_packages: str) -> tuple[str, ...]:
    packages = list(task_packages or BASE_TASK_PACKAGES)

    for task_package in _EXTERNAL_TASK_PACKAGES:
        if task_package not in packages:
            packages.append(task_package)

    return tuple(packages)
