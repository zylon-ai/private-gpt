from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from arq.worker import func

if TYPE_CHECKING:
    from arq.worker import Function

TaskFn = TypeVar("TaskFn", bound=Callable[..., Any])


def arq_task(*, name: str, max_tries: int = 1) -> Callable[[TaskFn], TaskFn]:
    def decorator(task_fn: TaskFn) -> TaskFn:
        _fn: Any = cast(Any, task_fn)
        _fn._arq_task_name = name
        _fn._arq_task_max_tries = max_tries
        return task_fn

    return decorator


def autodiscover_tasks(package_name: str = __name__) -> list[Function]:
    package = importlib.import_module(package_name)
    discovered: list[Function] = []

    for module_info in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        module = importlib.import_module(module_info.name)
        for value in vars(module).values():
            task_name = getattr(value, "_arq_task_name", None)
            if task_name is None:
                continue
            max_tries = cast(int, getattr(value, "_arq_task_max_tries", 1))
            discovered.append(
                func(
                    cast(Callable[..., Any], value), name=task_name, max_tries=max_tries
                )
            )

    return discovered
