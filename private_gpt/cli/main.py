import difflib
import sys

import typer

from private_gpt.cli.commands.run import run_command
from private_gpt.cli.commands.serve import serve_command

try:
    import celery as _celery  # noqa: F401

    from private_gpt.cli.commands.worker import worker_command

    _CELERY_AVAILABLE = True
except ImportError:
    _CELERY_AVAILABLE = False

app = typer.Typer(
    name="private-gpt",
    help="Private GPT server CLI — manage the server, workers, and connected apps.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)

app.command("serve")(serve_command)
if _CELERY_AVAILABLE:
    app.command("worker")(worker_command)
app.command(
    "run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(run_command)


@app.command("help")
def help_command(
    command: str = typer.Argument(
        "", show_default=False, help="Command to show help for"
    ),
) -> None:
    """Show help for any command."""
    if command:
        sys.argv = ["private-gpt", command, "--help"]
        app(standalone_mode=False)
    else:
        typer.echo(app.info.help or "")
        raise SystemExit(0)


_KNOWN_COMMANDS = ["serve", "run", "help"]
if _CELERY_AVAILABLE:
    _KNOWN_COMMANDS.append("worker")


def main() -> None:
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if not cmd.startswith("-") and cmd not in _KNOWN_COMMANDS:
            matches = difflib.get_close_matches(cmd, _KNOWN_COMMANDS, n=1, cutoff=0.6)
            if matches:
                typer.echo(
                    f"Unknown command: {cmd!r}. Did you mean: {matches[0]!r}?",
                    err=True,
                )
            else:
                typer.echo(f"Unknown command: {cmd!r}", err=True)
            raise SystemExit(1)
    app()


if __name__ == "__main__":
    main()
