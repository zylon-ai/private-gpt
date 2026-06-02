from pydantic import BaseModel


class CeleryErrorDetails(BaseModel):
    errors: list[str] | None = None
    warnings: list[str] | None = None


class CeleryError(ValueError):
    def __init__(
        self, errors: list[str] | None = None, warnings: list[str] | None = None
    ):
        self.details = CeleryErrorDetails(errors=errors, warnings=warnings)

    def dict(self) -> CeleryErrorDetails:
        return self.details

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(errors={self.details.errors}, warnings={self.details.warnings})"
