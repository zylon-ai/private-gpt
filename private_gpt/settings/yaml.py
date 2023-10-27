import os
import re
import typing
from typing import Any, TextIO

from yaml import SafeLoader

_env_replace_matcher = re.compile(r"\$\{(\w|_)+:?.*}")


@typing.no_type_check  # pyaml does not have good hints, everything is Any
def load_yaml_with_envvars(
    stream: TextIO, environ: dict[str, Any] = os.environ
) -> dict[str, Any]:
    """Load yaml file with environment variable expansion.

    The pattern ${VAR} or ${VAR:default} will be replaced with
    the value of the environment variable.
    """
    loader = SafeLoader(stream)

    def load_env_var(_, node) -> str:
        """Extract the matched value, expand env variable, and replace the match."""
        value = str(node.value).removeprefix("${").removesuffix("}")
        split = value.split(":", 1)
        env_var = split[0]
        value = environ.get(env_var)
        default = None if len(split) == 1 else split[1]
        if value is None and default is None:
            raise ValueError(
                f"Environment variable {env_var} is not set and not default was provided"
            )
        return value or default

    loader.add_implicit_resolver("env_var_replacer", _env_replace_matcher, None)
    loader.add_constructor("env_var_replacer", load_env_var)

    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
