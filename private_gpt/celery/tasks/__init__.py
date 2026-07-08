from private_gpt.celery.tasks.chat import __all__ as chat_all
from private_gpt.celery.tasks.ingestion import __all__ as ingestion_all

__all__ = []
__all__.extend(ingestion_all)
__all__.extend(chat_all)
