from collections.abc import Awaitable, Callable
from pathlib import Path

from injector import inject, singleton

from private_gpt.components.sandbox.content_bundle import (
    BundledFile,
    ContentBundle,
    StoredBundle,
)
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillVersionEntity,
)
from private_gpt.components.skills.paths import skill_mount_path
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.storage.storage_component import StorageComponent
from private_gpt.settings.settings import Settings


@singleton
class SkillLoader:
    """Resolves active skills from a SkillFilter into StoredBundles.

    Skill-specific knowledge (where skills live in storage, how to identify them)
    is encapsulated here. The environment layer receives generic bundles and
    does not need to know they came from skills.

    Bundles are references (storage prefix + lazy fetch): when the execution
    host can see the storage (s3fs mount or local storage dir), the skill is
    bind-mounted directly with no copying; fetch() is only invoked by the
    mounter's copy fallback.
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

    def bundles_for_versions(
        self, versions: list[SkillVersionEntity]
    ) -> list[ContentBundle]:
        """Create by-reference bundles from already-resolved skill version entities.

        Used by the skills interceptor to populate ``ContentBundlesLayer`` without
        a redundant ``recover_versions`` round-trip.
        """
        bundles: list[ContentBundle] = []
        for v in versions:
            bundles.append(
                StoredBundle(
                    canonical_path=skill_mount_path(v.skill_id),
                    storage_prefix=v.storage_prefix,
                    writable=False,
                    fetch=self._fetcher(v.storage_prefix),
                )
            )
        return bundles

    async def resolve(self, skill_filter: SkillFilter) -> list[StoredBundle]:
        """Resolve active skills into by-reference bundles. No downloads here.

        Each bundle points at the skill version's storage prefix and is mounted
        at canonical_path="/mnt/skills/{skill_id}/"; bytes are fetched lazily
        and only when the mounter cannot bind the storage path directly.
        """
        versions = await self._skill_service.recover_versions(skill_filter)
        return [
            StoredBundle(
                canonical_path=skill_mount_path(item.skill.id),
                storage_prefix=item.version.storage_prefix,
                writable=False,
                fetch=self._fetcher(item.version.storage_prefix),
            )
            for item in versions
        ]

    def _fetcher(self, prefix: str) -> Callable[[], Awaitable[list[BundledFile]]]:
        async def fetch() -> list[BundledFile]:
            file_paths = await self._storage.list_files(prefix)
            return [
                BundledFile(
                    path=fp,
                    content=await self._storage.read_file(prefix, fp),
                    permissions=0o444,
                )
                for fp in file_paths
            ]

        return fetch
