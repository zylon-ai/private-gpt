from typing import ClassVar

from private_gpt.celery.notify import ProgressStatus, ProgressStep


class IngestionProgressSteps(ProgressStep):
    """Possible steps for the ingestion progress."""

    VALIDATION = "Validation"
    PARSE = "Parse"
    STORAGE = "Storage"


class ValidationProgressStatus(ProgressStatus):
    current_step: ClassVar[IngestionProgressSteps] = IngestionProgressSteps.VALIDATION


class ParseProgressStatus(ProgressStatus):
    current_step: ClassVar[IngestionProgressSteps] = IngestionProgressSteps.PARSE


class StorageProgressStatus(ProgressStatus):
    current_step: ClassVar[IngestionProgressSteps] = IngestionProgressSteps.STORAGE
