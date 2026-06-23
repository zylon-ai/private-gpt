from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from injector import inject, singleton

from private_gpt.components.skills.models.skill_entities import (
    SkillEntity,
    SkillFilter,
    SkillFrontmatter,
    SkillVersionEntity,
    SkillVersionWithSkillEntity,
)
from private_gpt.components.skills.parser import (
    ParsedSkillDocument,
    parse_skill_markdown,
)
from private_gpt.components.skills.paths import skill_path
from private_gpt.components.skills.repositories.skill_repository import (
    CreateSkillInput,
    SkillPage,
    SkillPagination,
    SkillVersionPage,
    SQLAlchemySkillRepository,
    new_skill_id,
    new_skill_version_id,
    new_version_token,
)
from private_gpt.components.storage.models import StoredFile
from private_gpt.components.storage.storage_component import StorageComponent
from private_gpt.settings.settings import Settings


@singleton
class SkillService:
    @inject
    def __init__(
        self,
        settings: Settings,
        storage_component: StorageComponent,
        skill_repository: SQLAlchemySkillRepository,
    ) -> None:
        self._skill_repository = skill_repository
        self._max_bundle_size_bytes = settings.skills.max_bundle_size_bytes
        local_root = str(Path(settings.data.local_data_folder) / "storage")
        self._storage_bucket_name = settings.s3.durable_bucket_name
        self._storage_component = storage_component.get_object_storage(
            provider=settings.skills.storage_provider,
            local_root_path=local_root,
            bucket_name=self._storage_bucket_name,
        )

    def _check_bundle_size(self, files: list[StoredFile]) -> None:
        if self._max_bundle_size_bytes is None:
            return
        total = sum(len(f.content) for f in files)
        if total > self._max_bundle_size_bytes:
            raise ValueError(
                f"Skill bundle size ({total} bytes) exceeds the maximum allowed "
                f"size of {self._max_bundle_size_bytes} bytes."
            )

    async def create_skill(
        self,
        collection: str,
        display_title: str,
        source: Literal["custom", "anthropic", "zylon"],
        loading: Literal["eager", "lazy"],
        readonly: bool,
        files: list[StoredFile],
    ) -> SkillEntity:
        self._check_bundle_size(files)
        skill_id = new_skill_id()
        version_id = new_skill_version_id()
        now = datetime.now(tz=UTC)
        skill_markdown = _extract_skill_md(files)
        parsed_document = parse_skill_markdown(skill_markdown)

        frontmatter = SkillFrontmatter(
            name=parsed_document.frontmatter.name,
            description=parsed_document.frontmatter.description,
            license=parsed_document.frontmatter.license,
            compatibility=parsed_document.frontmatter.compatibility,
            metadata=parsed_document.frontmatter.metadata,
            allowed_tools=parsed_document.frontmatter.allowed_tools,
        )

        version_token = new_version_token()
        storage_prefix = skill_path(
            collection=collection, skill_id=skill_id, version_id=version_id
        )
        await self._storage_component.write_bundle(storage_prefix, files)

        initial_version = SkillVersionEntity(
            id=version_id,
            skill_id=skill_id,
            version=version_token,
            frontmatter=frontmatter,
            storage_prefix=storage_prefix,
            created_at=now,
        )

        try:
            return await self._skill_repository.create_skill(
                CreateSkillInput(
                    id=skill_id,
                    collection=collection,
                    display_title=display_title,
                    source=source,
                    loading=loading,
                    readonly=readonly,
                    created_at=now,
                    initial_version=initial_version,
                )
            )
        except Exception:
            await self._storage_component.delete_prefix(storage_prefix)
            raise

    async def validate_skill(self, files: list[StoredFile]) -> ParsedSkillDocument:
        """Dry-run: parse and validate files without persisting anything."""
        self._check_bundle_size(files)
        skill_markdown = _extract_skill_md(files)
        return parse_skill_markdown(skill_markdown)

    async def list_skills(
        self, collection: str, limit: int, page: str | None
    ) -> SkillPage:
        return await self._skill_repository.list_skills(
            collection=collection,
            pagination=SkillPagination(limit=limit, page=page),
        )

    async def get_skill(self, skill_id: str, collection: str) -> SkillEntity | None:
        return await self._skill_repository.get_skill(
            skill_id=skill_id, collection=collection
        )

    async def delete_skill(self, skill_id: str, collection: str) -> bool:
        versions = await self._skill_repository.list_versions(
            skill_id=skill_id,
            collection=collection,
            pagination=SkillPagination(limit=1000, page=None),
        )
        deleted = await self._skill_repository.delete_skill(
            skill_id=skill_id, collection=collection
        )
        if deleted and versions is not None:
            for version in versions.data:
                await self._storage_component.delete_prefix(version.storage_prefix)
        return deleted

    async def create_version(
        self,
        skill_id: str,
        collection: str,
        files: list[StoredFile],
    ) -> SkillVersionEntity | None:
        self._check_bundle_size(files)
        now = datetime.now(tz=UTC)
        skill_markdown = _extract_skill_md(files)
        parsed_document = parse_skill_markdown(skill_markdown)
        frontmatter = SkillFrontmatter(
            name=parsed_document.frontmatter.name,
            description=parsed_document.frontmatter.description,
            license=parsed_document.frontmatter.license,
            compatibility=parsed_document.frontmatter.compatibility,
            metadata=parsed_document.frontmatter.metadata,
            allowed_tools=parsed_document.frontmatter.allowed_tools,
        )

        version_id = new_skill_version_id()
        version_token = new_version_token()
        storage_prefix = skill_path(
            collection=collection, skill_id=skill_id, version_id=version_id
        )
        await self._storage_component.write_bundle(storage_prefix, files)
        created_version = SkillVersionEntity(
            id=version_id,
            skill_id=skill_id,
            version=version_token,
            frontmatter=frontmatter,
            storage_prefix=storage_prefix,
            created_at=now,
        )

        try:
            saved = await self._skill_repository.create_version(
                created_version, collection=collection
            )
        except Exception:
            await self._storage_component.delete_prefix(storage_prefix)
            raise
        if saved is None:
            await self._storage_component.delete_prefix(storage_prefix)
        return saved

    async def list_versions(
        self,
        skill_id: str,
        collection: str,
        limit: int,
        page: str | None,
    ) -> SkillVersionPage:
        versions = await self._skill_repository.list_versions(
            skill_id=skill_id,
            collection=collection,
            pagination=SkillPagination(limit=limit, page=page),
        )
        return SkillVersionPage(
            data=versions.data if versions else [],
            has_more=versions.has_more if versions else False,
            next_page=versions.next_page if versions else None,
        )

    async def get_version(
        self,
        skill_id: str,
        version: str,
        collection: str,
    ) -> SkillVersionEntity | None:
        found = await self._skill_repository.get_version(
            skill_id=skill_id,
            version=version,
            collection=collection,
        )
        return found

    async def delete_version(
        self, skill_id: str, version: str, collection: str
    ) -> bool:
        item = await self._skill_repository.get_version(
            skill_id=skill_id,
            version=version,
            collection=collection,
        )
        deleted = await self._skill_repository.delete_version(
            skill_id=skill_id,
            version=version,
            collection=collection,
        )
        if deleted and item is not None:
            await self._storage_component.delete_prefix(item.storage_prefix)
        return deleted

    async def recover_versions(
        self, skill_filter: SkillFilter
    ) -> list[SkillVersionWithSkillEntity]:
        versions = await self._skill_repository.recover_versions(
            skill_filter=skill_filter
        )
        skills_page = await self._skill_repository.list_skills(
            collection=skill_filter.collection,
            pagination=SkillPagination(limit=1000, page=None),
        )
        skills_by_id = {item.id: item for item in skills_page.data}

        resolved: list[SkillVersionWithSkillEntity] = []
        for version in versions:
            skill = skills_by_id.get(version.skill_id)
            if not skill:
                continue
            resolved.append(
                SkillVersionWithSkillEntity(
                    skill=skill,
                    version=version,
                )
            )
        return resolved

    async def get_skill_body(self, version: SkillVersionEntity) -> str:
        markdown_bytes = await self._storage_component.read_file(
            version.storage_prefix, "SKILL.md"
        )
        parsed = parse_skill_markdown(markdown_bytes.decode("utf-8"))
        return parsed.body


def _extract_skill_md(files: list[StoredFile]) -> str:
    for file in files:
        if file.path == "SKILL.md":
            return file.content.decode("utf-8")
    raise ValueError("SKILL.md is required")
