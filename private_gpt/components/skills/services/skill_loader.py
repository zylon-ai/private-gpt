from pathlib import Path

from injector import inject, singleton

from private_gpt.components.code_execution.content_bundle import (
    BundledFile,
    ContentBundle,
)
from private_gpt.components.skills.models.skill_entities import SkillFilter
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.storage.storage_component import StorageComponent
from private_gpt.settings.settings import Settings


@singleton
class SkillLoader:
    """Resolves active skills from a SkillFilter into ContentBundles.

    Skill-specific knowledge (where skills live in storage, how to identify them)
    is encapsulated here. The code execution layer receives generic ContentBundles
    and does not need to know they came from skills.

    Future content types (plugins, etc.) follow the same pattern with their own
    loader, returning ContentBundles with different canonical_path prefixes.
    """

    @inject
    def __init__(
        self,
        settings: Settings,
        storage_component: StorageComponent,
        skill_service: SkillService,
    ) -> None:
        self._skill_service = skill_service
        local_root = str(Path(settings.data.local_data_folder) / "storage")
        self._storage = storage_component.get_object_storage(
            provider=settings.skills.storage_provider,
            local_root_path=local_root,
            bucket_name=settings.s3.durable_bucket_name,
        )

    async def load(self, skill_filter: SkillFilter) -> list[ContentBundle]:
        """Download skill files from object storage and return them as ContentBundles.

        Each bundle has canonical_path="/mnt/skills/{skill_id}/" and the raw bytes
        of every file stored under that skill version's storage prefix.
        """
        versions = await self._skill_service.recover_versions(skill_filter)
        bundles: list[ContentBundle] = []
        for item in versions:
            prefix = item.version.storage_prefix
            file_paths = await self._storage.list_files(prefix)
            files = {fp: await self._storage.read_file(prefix, fp) for fp in file_paths}
            bundles.append(
                ContentBundle(
                    canonical_path=f"/mnt/skills/{item.skill.id}/",
                    files=[
                        BundledFile(path=fp, content=content, permissions=0o444)
                        for fp, content in files.items()
                    ],
                    writable=False,
                )
            )
        return bundles
