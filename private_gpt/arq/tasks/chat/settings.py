from private_gpt.arq.settings import get_queue_name as build_queue_name
from private_gpt.settings.settings import Settings

START_CHAT_TASK_NAME = "private_gpt.chat.start"
RESUME_ITERATION_TASK_NAME = "private_gpt.chat.resume_iteration"
TOOL_RESUME_TASK_NAME = "private_gpt.tool.resume"
TOOL_TIMEOUT_TASK_NAME = "private_gpt.tool.timeout"


def get_queue_name(settings: Settings) -> str:
    return build_queue_name(settings.scheduler.chat.celery_queue)
