import enum


class ToolValidationMode(enum.StrEnum):
    EAGER = "eager"
    LAZY = "lazy"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, mode_str: str) -> "ToolValidationMode":
        """Create a ToolValidationMode from a string."""
        mode_str = mode_str.lower()
        if mode_str == "eager":
            return cls.EAGER
        elif mode_str == "lazy":
            return cls.LAZY
        else:
            raise ValueError(f"Invalid ToolValidationMode: {mode_str}")
