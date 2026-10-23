from private_gpt.arq.task_registry import get_task_packages
from private_gpt.arq.tasks import autodiscover_registered_tasks


def test_explicit_task_packages_replace_defaults() -> None:
    assert get_task_packages("private_gpt.arq.tasks.chat") == (
        "private_gpt.arq.tasks.chat",
    )


def test_chat_package_discovers_only_chat_tasks() -> None:
    assert {
        task.name
        for task in autodiscover_registered_tasks("private_gpt.arq.tasks.chat")
    } == {
        "private_gpt.chat.resume_iteration",
        "private_gpt.chat.start",
        "private_gpt.tool.resume",
        "private_gpt.tool.timeout",
    }
