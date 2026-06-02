from collections.abc import Sequence


def format_missing_dependency_message(
    feature: str,
    *,
    extras: str | Sequence[str] | None = None,
    keep_other_deps: bool = True,
) -> str:
    message = f"{feature} dependencies are not installed."

    if extras is None:
        return message

    command_prefix = (
        "uv sync --inexact --extra" if keep_other_deps else "uv sync --extra"
    )

    if isinstance(extras, str):
        return f"{message} Install with `{command_prefix} {extras}`."

    commands = [f"`{command_prefix} {extra}`" for extra in extras]
    if len(commands) == 1:
        return f"{message} Install with {commands[0]}."

    install_options = " or ".join(commands)
    return f"{message} Install with one of: {install_options}."
