import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Literal, cast

from injector import inject, singleton
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Subquery

from private_gpt.components.persistence.persistence_component import (
    PersistenceComponent,
)
from private_gpt.components.persistence.repository_base import SQLAlchemyRepositoryBase
from private_gpt.components.skills.models.skill_entities import (
    SkillEntity,
    SkillFilter,
    SkillFrontmatter,
    SkillVersionEntity,
)
from private_gpt.components.skills.models.sqlalchemy_models import (
    SkillORM,
    SkillVersionORM,
)
from private_gpt.settings.settings import Settings


class SkillPagination(BaseModel):
    limit: int = Field(
        description="Maximum records in this page.",
        ge=1,
        le=1000,
    )
    page: str | None = Field(
        default=None,
        description="Opaque page token (offset-based implementation).",
    )


class SkillPage(BaseModel):
    data: list[SkillEntity] = Field(default_factory=list, description="Page items.")
    has_more: bool = Field(description="Whether next page is available.")
    next_page: str | None = Field(default=None, description="Token for next page.")


class SkillVersionPage(BaseModel):
    data: list[SkillVersionEntity] = Field(
        default_factory=list,
        description="Page of skill versions.",
    )
    has_more: bool = Field(description="Whether next page is available.")
    next_page: str | None = Field(default=None, description="Token for next page.")


class CreateSkillInput(BaseModel):
    id: str = Field(description="Generated skill identifier.")
    collection: str = Field(description="Tenant collection identifier.", min_length=1)
    display_title: str = Field(description="Human display title.", min_length=1)
    source: Literal["custom", "anthropic", "zylon"] = Field(
        description="Skill source provider."
    )
    loading: Literal["eager", "lazy"] = Field(description="Skill loading mode.")
    readonly: bool = Field(default=False, description="Readonly flag.")
    created_at: datetime = Field(description="Creation timestamp.")
    initial_version: SkillVersionEntity = Field(
        description="Initial skill version to persist together with skill."
    )


class SkillRepository(ABC):
    @abstractmethod
    async def create_skill(self, payload: CreateSkillInput) -> SkillEntity: ...

    @abstractmethod
    async def get_skill(self, skill_id: str, collection: str) -> SkillEntity | None: ...

    @abstractmethod
    async def list_skills(
        self, collection: str, pagination: SkillPagination
    ) -> SkillPage: ...

    @abstractmethod
    async def delete_skill(self, skill_id: str, collection: str) -> bool: ...

    @abstractmethod
    async def create_version(
        self, payload: SkillVersionEntity, collection: str
    ) -> SkillVersionEntity | None: ...

    @abstractmethod
    async def get_version(
        self,
        skill_id: str,
        version: str,
        collection: str,
    ) -> SkillVersionEntity | None: ...

    @abstractmethod
    async def list_versions(
        self,
        skill_id: str,
        collection: str,
        pagination: SkillPagination,
    ) -> SkillVersionPage | None: ...

    @abstractmethod
    async def delete_version(
        self, skill_id: str, version: str, collection: str
    ) -> bool: ...

    @abstractmethod
    async def recover_versions(
        self, skill_filter: SkillFilter
    ) -> list[SkillVersionEntity]: ...


@singleton
class SQLAlchemySkillRepository(SkillRepository, SQLAlchemyRepositoryBase):
    @inject
    def __init__(
        self, settings: Settings, persistence_component: PersistenceComponent
    ) -> None:
        super().__init__(
            persistence_component=persistence_component,
            store=settings.skills.database,
        )

    async def create_skill(self, payload: CreateSkillInput) -> SkillEntity:
        async with self._session_factory() as session:
            skill = SkillORM(
                id=payload.id,
                collection=payload.collection,
                display_title=payload.display_title,
                source=payload.source,
                loading=payload.loading,
                readonly=payload.readonly,
                created_at=payload.created_at,
                updated_at=payload.created_at,
            )
            session.add(skill)
            await session.flush()

            version = SkillVersionORM(
                id=payload.initial_version.id,
                skill_id=payload.id,
                version=payload.initial_version.version,
                frontmatter_json=payload.initial_version.frontmatter.model_dump(
                    mode="json"
                ),
                storage_prefix=payload.initial_version.storage_prefix,
                created_at=payload.initial_version.created_at,
                updated_at=payload.initial_version.created_at,
            )
            session.add(version)
            await session.commit()
            await session.refresh(skill)
            return _skill_from_orm(
                skill,
                latest_version=payload.initial_version.version,
            )

    async def get_skill(self, skill_id: str, collection: str) -> SkillEntity | None:
        async with self._session_factory() as session:
            query = select(SkillORM).where(
                SkillORM.id == skill_id,
                SkillORM.collection == collection,
            )
            row = (await session.scalars(query)).first()
            if row is None:
                return None
            latest = await _latest_version_row_for_skill(session, row.id)
            return _skill_from_orm(
                row,
                latest_version=latest.version if latest else None,
            )

    async def list_skills(
        self, collection: str, pagination: SkillPagination
    ) -> SkillPage:
        offset = _parse_page(pagination.page)
        async with self._session_factory() as session:
            query = (
                select(SkillORM)
                .where(SkillORM.collection == collection)
                .order_by(desc(SkillORM.created_at), desc(SkillORM.id))
            )
            rows = list(
                (
                    await session.scalars(
                        query.offset(offset).limit(pagination.limit + 1)
                    )
                ).all()
            )

            has_more = len(rows) > pagination.limit
            page_rows = rows[: pagination.limit]
            items: list[SkillEntity] = []
            for row in page_rows:
                latest = await _latest_version_row_for_skill(session, row.id)
                items.append(
                    _skill_from_orm(
                        row,
                        latest_version=latest.version if latest else None,
                    )
                )

        next_page = str(offset + pagination.limit) if has_more else None
        return SkillPage(data=items, has_more=has_more, next_page=next_page)

    async def delete_skill(self, skill_id: str, collection: str) -> bool:
        async with self._session_factory() as session:
            query = select(SkillORM).where(
                SkillORM.id == skill_id,
                SkillORM.collection == collection,
            )
            row = (await session.scalars(query)).first()
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def create_version(
        self, payload: SkillVersionEntity, collection: str
    ) -> SkillVersionEntity | None:
        async with self._session_factory() as session:
            skill_query = select(SkillORM).where(
                SkillORM.id == payload.skill_id,
                SkillORM.collection == collection,
            )
            skill = (await session.scalars(skill_query)).first()
            if skill is None:
                return None

            version = SkillVersionORM(
                id=payload.id,
                skill_id=payload.skill_id,
                version=payload.version,
                frontmatter_json=payload.frontmatter.model_dump(mode="json"),
                storage_prefix=payload.storage_prefix,
                created_at=payload.created_at,
                updated_at=payload.created_at,
            )
            session.add(version)
            skill.updated_at = datetime.now(tz=UTC)
            await session.commit()
            return payload

    async def get_version(
        self, skill_id: str, version: str, collection: str
    ) -> SkillVersionEntity | None:
        async with self._session_factory() as session:
            if not await self._skill_exists(session, skill_id, collection):
                return None
            query = select(SkillVersionORM).where(
                SkillVersionORM.skill_id == skill_id,
                SkillVersionORM.version == version,
            )
            row = (await session.scalars(query)).first()
            return _version_from_orm(row) if row else None

    async def list_versions(
        self,
        skill_id: str,
        collection: str,
        pagination: SkillPagination,
    ) -> SkillVersionPage | None:
        offset = _parse_page(pagination.page)
        async with self._session_factory() as session:
            if not await self._skill_exists(session, skill_id, collection):
                return None
            query = (
                select(SkillVersionORM)
                .where(SkillVersionORM.skill_id == skill_id)
                .order_by(desc(SkillVersionORM.created_at), desc(SkillVersionORM.id))
            )
            rows = list(
                (
                    await session.scalars(
                        query.offset(offset).limit(pagination.limit + 1)
                    )
                ).all()
            )

        has_more = len(rows) > pagination.limit
        page_rows = rows[: pagination.limit]
        next_page = str(offset + pagination.limit) if has_more else None
        return SkillVersionPage(
            data=[_version_from_orm(row) for row in page_rows],
            has_more=has_more,
            next_page=next_page,
        )

    async def delete_version(
        self, skill_id: str, version: str, collection: str
    ) -> bool:
        async with self._session_factory() as session:
            skill_query = select(SkillORM).where(
                SkillORM.id == skill_id,
                SkillORM.collection == collection,
            )
            skill = (await session.scalars(skill_query)).first()
            if skill is None:
                return False

            version_query = select(SkillVersionORM).where(
                SkillVersionORM.skill_id == skill_id,
                SkillVersionORM.version == version,
            )
            version_row = (await session.scalars(version_query)).first()
            if version_row is None:
                return False

            await session.delete(version_row)
            skill.updated_at = datetime.now(tz=UTC)
            await session.commit()
            return True

    async def recover_versions(
        self, skill_filter: SkillFilter
    ) -> list[SkillVersionEntity]:
        async with self._session_factory() as session:
            identifiers = skill_filter.skill_or_version_ids
            if identifiers is None:
                return await self._recover_all_latest_versions(session, skill_filter)
            if not identifiers:
                return []
            return await self._recover_by_identifiers(
                session, set(identifiers), skill_filter.collection
            )

    async def _recover_by_identifiers(
        self,
        session: AsyncSession,
        identifiers: set[str],
        collection: str,
    ) -> list[SkillVersionEntity]:
        skills = {
            row.id: row
            for row in (
                await session.scalars(
                    select(SkillORM).where(
                        SkillORM.collection == collection,
                        SkillORM.id.in_(identifiers),
                    )
                )
            ).all()
        }

        versions = {
            row.id: row
            for row in (
                await session.scalars(
                    select(SkillVersionORM)
                    .join(SkillORM, SkillVersionORM.skill_id == SkillORM.id)
                    .where(
                        SkillORM.collection == collection,
                        SkillVersionORM.id.in_(identifiers),
                    )
                )
            ).all()
        }

        missing = sorted(identifiers - skills.keys() - versions.keys())
        if missing:
            raise ValueError(
                f"Unknown skill_or_version_ids in collection '{collection}': {', '.join(missing)}"
            )

        result: list[SkillVersionEntity] = [
            _version_from_orm(row) for row in versions.values()
        ]

        for skill_id in skills:
            latest = await _latest_version_row_for_skill(session, skill_id)
            if latest:
                result.append(_version_from_orm(latest))

        return result

    async def _recover_all_latest_versions(
        self, session: AsyncSession, skill_filter: SkillFilter
    ) -> list[SkillVersionEntity]:
        latest_versions = _latest_versions_subquery()
        query = (
            select(
                latest_versions.c.version_id,
                latest_versions.c.skill_id,
                latest_versions.c.version,
                latest_versions.c.frontmatter_json,
                latest_versions.c.storage_prefix,
                latest_versions.c.created_at,
            )
            .select_from(SkillORM)
            .join(
                latest_versions,
                and_(
                    latest_versions.c.skill_id == SkillORM.id,
                    latest_versions.c.rn == 1,
                ),
            )
            .where(SkillORM.collection == skill_filter.collection)
            .order_by(desc(SkillORM.created_at), desc(SkillORM.id))
        )
        rows = (await session.execute(query)).mappings().all()
        return [
            _version_from_values(
                version_id=cast(str, row["version_id"]),
                skill_id=cast(str, row["skill_id"]),
                version=cast(str, row["version"]),
                frontmatter_json=cast(dict[str, object], row["frontmatter_json"]),
                storage_prefix=cast(str, row["storage_prefix"]),
                created_at=cast(datetime, row["created_at"]),
            )
            for row in rows
        ]

    async def _skill_exists(
        self, session: AsyncSession, skill_id: str, collection: str
    ) -> bool:
        query = select(SkillORM.id).where(
            SkillORM.id == skill_id,
            SkillORM.collection == collection,
        )
        return (await session.execute(query)).first() is not None


def new_skill_id() -> str:
    return f"skill_{uuid.uuid4().hex}"


def new_skill_version_id() -> str:
    return f"skillver_{uuid.uuid4().hex}"


def new_version_token() -> str:
    return str(time.time_ns() // 1_000)


def _parse_page(page: str | None) -> int:
    if page is None:
        return 0
    value = int(page)
    if value < 0:
        raise ValueError("Invalid page token")
    return value


def _skill_from_orm(
    row: SkillORM,
    latest_version: str | None,
) -> SkillEntity:
    source = cast(Literal["custom", "anthropic", "zylon"], row.source)
    loading = cast(Literal["eager", "lazy"], row.loading)
    return SkillEntity(
        id=row.id,
        collection=row.collection,
        display_title=row.display_title,
        source=source,
        loading=loading,
        readonly=row.readonly,
        latest_version=latest_version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _version_from_orm(row: SkillVersionORM) -> SkillVersionEntity:
    return _version_from_values(
        version_id=row.id,
        skill_id=row.skill_id,
        version=row.version,
        frontmatter_json=row.frontmatter_json,
        storage_prefix=row.storage_prefix,
        created_at=row.created_at,
    )


def _version_from_values(
    version_id: str,
    skill_id: str,
    version: str,
    frontmatter_json: dict[str, object],
    storage_prefix: str,
    created_at: datetime,
) -> SkillVersionEntity:
    frontmatter = SkillFrontmatter.model_validate(frontmatter_json)
    return SkillVersionEntity(
        id=version_id,
        skill_id=skill_id,
        version=version,
        frontmatter=frontmatter,
        storage_prefix=storage_prefix,
        created_at=created_at,
    )


async def _latest_version_row_for_skill(
    session: AsyncSession, skill_id: str
) -> SkillVersionORM | None:
    return (
        await session.scalars(
            select(SkillVersionORM)
            .where(SkillVersionORM.skill_id == skill_id)
            .order_by(desc(SkillVersionORM.created_at), desc(SkillVersionORM.id))
            .limit(1)
        )
    ).first()


def _latest_versions_subquery() -> Subquery:
    return select(
        SkillVersionORM.id.label("version_id"),
        SkillVersionORM.skill_id.label("skill_id"),
        SkillVersionORM.version.label("version"),
        SkillVersionORM.frontmatter_json.label("frontmatter_json"),
        SkillVersionORM.storage_prefix.label("storage_prefix"),
        SkillVersionORM.created_at.label("created_at"),
        func.row_number()
        .over(
            partition_by=SkillVersionORM.skill_id,
            order_by=(desc(SkillVersionORM.created_at), desc(SkillVersionORM.id)),
        )
        .label("rn"),
    ).subquery("latest_versions")
