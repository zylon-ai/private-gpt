from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from injector import inject, singleton

from private_gpt.arq.enqueue import abort_chat_job, enqueue_chat_job
from private_gpt.celery.dispatch import dispatch_task
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.stream.runtime import start_stream_processing
from private_gpt.components.streaming.stream_component import StreamComponent
from private_gpt.components.streaming.tasks.task_manager import TaskManager
from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.server.chat.chat_request_mapper import ChatRequestMapper
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.server.chat.chat_models import ChatBody

CHAT_TASK_NAME = "private_gpt.chat.run"


class BaseChatScheduler(ABC):
    @abstractmethod
    async def create(
        self,
        body: ChatBody,
        message_id: str | None = None,
    ) -> str:
        ...

    @abstractmethod
    async def cancel(self, correlation_id: str) -> bool:
        ...


@singleton
class LocalChatScheduler(BaseChatScheduler):
    @inject
    def __init__(
        self,
        chat_service: ChatService,
        chat_request_mapper: ChatRequestMapper,
        stream_component: StreamComponent,
        task_manager: TaskManager,
    ) -> None:
        self._chat_service = chat_service
        self._chat_request_mapper = chat_request_mapper
        self._stream_service = stream_component.stream
        self._task_manager = task_manager

    async def create(
        self,
        body: ChatBody,
        message_id: str | None = None,
    ) -> str:
        request = await self._chat_request_mapper.create_request_from_body(body)
        completion_gen = await self._chat_service.stream_chat(request)
        metadata = {"message_count": len(body.messages)}

        correlation_id = await self._stream_service.create_stream(
            stream_type="chat_completion",
            correlation_id=message_id,
            metadata=metadata,
        )

        await start_stream_processing(
            stream_service=self._stream_service,
            task_manager=self._task_manager,
            correlation_id=correlation_id,
            stream_type="chat_completion",
            event_generator=completion_gen.events,  # type: ignore[arg-type]
            event_handler=StreamingEventHandler(),
            metadata=metadata,
        )
        return correlation_id

    async def cancel(self, correlation_id: str) -> bool:
        del correlation_id
        return False


@singleton
class CeleryChatScheduler(BaseChatScheduler):
    @inject
    def __init__(self, settings: Settings, stream_component: StreamComponent) -> None:
        self._settings = settings
        self._stream_service = stream_component.stream

    async def create(
        self,
        body: ChatBody,
        message_id: str | None = None,
    ) -> str:
        metadata: dict[str, Any] = {"message_count": len(body.messages)}

        correlation_id = await self._stream_service.create_stream(
            stream_type="chat_completion",
            correlation_id=message_id,
            metadata=metadata,
        )

        try:
            dispatch_task(
                task_name=CHAT_TASK_NAME,
                args=[body, correlation_id, "chat_completion", metadata],
                queue=self._settings.scheduler.chat.celery_queue,
                task_id=correlation_id,
            )
        except Exception as exc:
            await self._stream_service.update_stream_status(
                correlation_id,
                StreamStatus.ERROR,
                error_message=str(exc),
                metadata=metadata,
            )
            raise

        return correlation_id

    async def cancel(self, correlation_id: str) -> bool:
        from private_gpt.celery.celery import celery_app

        celery_app.control.revoke(correlation_id, terminate=False)
        return True


@singleton
class ArqChatScheduler(BaseChatScheduler):
    @inject
    def __init__(self, stream_component: StreamComponent) -> None:
        self._stream_service = stream_component.stream

    async def create(
        self,
        body: ChatBody,
        message_id: str | None = None,
    ) -> str:
        metadata: dict[str, Any] = {"message_count": len(body.messages)}

        correlation_id = await self._stream_service.create_stream(
            stream_type="chat_completion",
            correlation_id=message_id,
            metadata=metadata,
        )

        try:
            await enqueue_chat_job(
                body=body.model_dump(mode="json"),
                correlation_id=correlation_id,
                stream_type="chat_completion",
                metadata=metadata,
            )
        except Exception as exc:
            await self._stream_service.update_stream_status(
                correlation_id,
                StreamStatus.ERROR,
                error_message=str(exc),
                metadata=metadata,
            )
            raise

        return correlation_id

    async def cancel(self, correlation_id: str) -> bool:
        return await abort_chat_job(correlation_id=correlation_id)


@singleton
class ChatSchedulerFactory:
    @inject
    def __init__(
        self,
        settings: Settings,
        local: LocalChatScheduler,
        celery: CeleryChatScheduler,
        arq: ArqChatScheduler,
    ) -> None:
        self._scheduler: BaseChatScheduler = {
            "local": local,
            "celery": celery,
            "arq": arq,
        }[settings.scheduler.chat.mode]

    def get(self) -> BaseChatScheduler:
        return self._scheduler
