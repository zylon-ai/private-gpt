import ast
import os
import subprocess
import sys

import pytest

from private_gpt.celery.task_registry import get_task_packages


def test_explicit_task_packages_replace_defaults() -> None:
    assert get_task_packages("private_gpt.celery.tasks.tools") == (
        "private_gpt.celery.tasks.tools",
    )


@pytest.mark.parametrize(
    ("task_package", "expected_tasks"),
    [
        (
            "private_gpt.celery.tasks.ingestion",
            {
                "private_gpt.ingestion.delete",
                "private_gpt.ingestion.vector_index",
            },
        ),
        (
            "private_gpt.celery.tasks.tools",
            {"private_gpt.tools.run"},
        ),
    ],
)
def test_worker_registers_only_configured_task_package(
    task_package: str,
    expected_tasks: set[str],
) -> None:
    env = os.environ.copy()
    env["PGPT_CELERY_TASK_PACKAGES"] = task_package
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from private_gpt.celery.celery import celery_app; "
            "celery_app.loader.import_default_modules(); "
            "print(sorted(name for name in celery_app.tasks "
            "if name.startswith('private_gpt.')))",
        ],
        env=env,
        text=True,
    )

    assert set(ast.literal_eval(output.strip())) == expected_tasks
