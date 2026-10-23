from private_gpt.celery.error import CeleryError


class InvalidFileError(CeleryError):
    pass


class NotControlledArtifactError(RuntimeError):
    pass


class ModelNotAvailableError(RuntimeError):
    pass
