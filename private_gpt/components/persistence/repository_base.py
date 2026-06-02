from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

from private_gpt.components.persistence.persistence_component import (
    PersistenceComponent,
)


class _DatabaseClient(Protocol):
    sync_session: sessionmaker[Session]
    async_session: async_sessionmaker[AsyncSession]


class SQLAlchemyRepositoryBase:
    def __init__(self, persistence_component: PersistenceComponent, store: str) -> None:
        self._persistence_component = persistence_component
        client = self._persistence_component.get_client(store=store)
        if not client:
            raise ValueError("You cannot use this client to generate session objects")
        self._client = client

    @asynccontextmanager
    async def _session_factory(self) -> AsyncIterator[AsyncSession]:
        async with self._client.async_session() as session:
            yield session
