from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

from injector import inject, singleton

from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry
from private_gpt.components.sandbox.mount import (
    Mount,
    MountFile,
    MountSource,
    UriSource,
)
from private_gpt.components.skills.models.skill_entities import (
    SkillFilter,
    SkillVersionEntity,
)
from private_gpt.components.skills.paths import skill_mount_path
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.components.storage.storage_component import StorageComponent
from private_gpt.settings.settings import Settings


def skill_directory_fetch(
    prefix: str, filename: str | None = None
) -> Callable[[], Awaitable[list[MountFile]]]:
    """Build a storage-aware fetch for a skill version directory.

    Referenced by ``UriSource.fetch_ref`` so the fetch survives JSON
    round-trips (tool configs are serialized through the task scheduler).
    """
    from private_gpt.di import get_global_injector
    from private_gpt.settings.settings import settings as current_settings

    storage = (
        get_global_injector()
        .get(StorageComponent)
        .get_object_storage(
            provider=current_settings().skills.storage_provider,
            local_root_path=str(
                Path(current_settings().data.local_data_folder) / "storage"
            ),
            bucket_name=current_settings().s3.durable_bucket_name,
        )
    )

    async def fetch() -> list[MountFile]:
        file_paths = await storage.list_files(prefix)
        return [
            MountFile(
                path=fp,
                content=await storage.read_file(prefix, fp),
                permissions=0o444,
            )
            for fp in file_paths
        ]

    return fetch


@singleton
class SkillLoader:
    """Resolves active skills from a SkillFilter into namespace-backed mounts.

    Every skill is a read-only **folder** mount at ``/mnt/skills/{name}/``,
    backed by the version directory under the ``skills`` namespace root
    (``{skills_root}/{storage_prefix}``). The URI source exists so the
    hydration layer can (re)fill that host folder when content is not already
    present on disk.
    """

    @inject
    def __init__(
        self,
        settings: Settings,
        storage_component: StorageComponent,
        skill_service: SkillService,
        namespace_registry: NamespaceRegistry,
    ) -> None:
        self._skill_service = skill_service
        self._namespace_registry = namespace_registry
        local_root = str(Path(settings.data.local_data_folder) / "storage")
        self._storage = storage_component.get_object_storage(
            provider=settings.skills.storage_provider,
            local_root_path=local_root,
            bucket_name=settings.s3.durable_bucket_name,
        )

    def mounts_for_versions(self, versions: list[SkillVersionEntity]) -> list[Mount]:
        """Create namespace-backed mounts from already-resolved skill versions.

        Each skill is a read-only folder mount at ``/mnt/skills/{name}/`` whose
        host folder is ``{skills_root}/{version.storage_prefix}`` so an update
        produces a fresh path (and a fresh hydration when enabled).
        """
        skills_root: Path | None = None
        with suppress(KeyError):
            skills_root = self._namespace_registry.root("skills")

        mounts: list[Mount] = []
        for version in versions:
            host_path = (skills_root / version.storage_prefix) if skills_root is not None else None
            mounts.append(
                Mount(
                    target=skill_mount_path(version.frontmatter.name),
                    access="ro",
                    name=f"skill:{version.frontmatter.name}",
                    host_path=host_path,
                    uri_source=UriSource(
                        uri=version.storage_prefix,
                        fetch=self._fetcher(version.storage_prefix),
                        fetch_ref=(
                            "private_gpt.components.skills.services.skill_loader"
                            ":skill_directory_fetch"
                        ),
                    ),
                    source=MountSource(
                        namespace="skills",
                        scope=version.id,
                        path="",
                    ),
                    etag=version.id,
                )
            )
        return mounts

    async def resolve(self, skill_filter: SkillFilter) -> list[Mount]:
        """Resolve active skills into namespace-backed mounts. No downloads here."""
        versions = await self._skill_service.recover_versions(skill_filter)
        return self.mounts_for_versions([item.version for item in versions])

    def _fetcher(self, prefix: str) -> Callable[[], Awaitable[list[MountFile]]]:
        async def fetch() -> list[MountFile]:
            file_paths = await self._storage.list_files(prefix)
            return [
                MountFile(
                    path=fp,
                    content=await self._storage.read_file(prefix, fp),
                    permissions=0o444,
                )
                for fp in file_paths
            ]

        return fetch
