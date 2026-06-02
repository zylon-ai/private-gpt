import io
import os

import pytest

from private_gpt.settings.settings_loader import discover_models_from_environment
from private_gpt.settings.yaml import load_yaml_with_envvars


def test_environment_variables_are_loaded() -> None:
    sample_yaml = """
    replaced: ${TEST_REPLACE_ME}
    """
    env = {"TEST_REPLACE_ME": "replaced"}
    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), env)
    os.environ.copy()
    assert loaded["replaced"] == "replaced"


def test_environment_defaults_variables_are_loaded() -> None:
    sample_yaml = """
    replaced: ${PGPT_EMBEDDING_HF_MODEL_NAME:BAAI/bge-small-en-v1.5}
    """
    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), {})
    assert loaded["replaced"] == "BAAI/bge-small-en-v1.5"


def test_environment_defaults_variables_are_loaded_with_duplicated_delimiters() -> None:
    sample_yaml = """
    replaced: ${PGPT_EMBEDDING_HF_MODEL_NAME::duped::}
    """
    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), {})
    assert loaded["replaced"] == ":duped::"


def test_environment_without_defaults_fails() -> None:
    sample_yaml = """
    replaced: ${TEST_REPLACE_ME}
    """
    with pytest.raises(ValueError) as error:
        load_yaml_with_envvars(io.StringIO(sample_yaml), {})
    assert error is not None


def test_load_models_from_env() -> None:
    env = {
        "PGPT_MODELS_MODEL3_NAME": "model3",
        "PGPT_MODELS_MODEL3_PARAM1": "value3",
        "PGPT_MODELS_MODEL3_PARAM2__SUBPARAM": "subvalue",
        "PGPT_MODELS_MODEL3_PARAM2__SUBPARAM2__SUBSUB": "subsubvalue",
    }
    loaded = discover_models_from_environment(env)
    assert len(loaded) == 1
    first_model = loaded[0]
    assert first_model["name"] == "model3"
    assert first_model["param1"] == "value3"
    assert first_model["param2"]["subparam"] == "subvalue"
    assert first_model["param2"]["subparam2"]["subsub"] == "subsubvalue"


def test_multiple_environment_variables_fallback() -> None:
    sample_yaml = """
    replaced: ${VAR1,VAR2,VAR3:default_value}
    """

    env = {"VAR1": "first"}
    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), env)
    assert loaded["replaced"] == "first"

    env = {"VAR2": "second"}
    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), env)
    assert loaded["replaced"] == "second"

    env = {"VAR2": "second", "VAR3": "third"}
    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), env)
    assert loaded["replaced"] == "second"

    loaded = load_yaml_with_envvars(io.StringIO(sample_yaml), {})
    assert loaded["replaced"] == "default_value"


def test_multiple_environment_variables_without_default_fails() -> None:
    sample_yaml = """
    replaced: ${VAR1,VAR2}
    """
    with pytest.raises(ValueError):
        load_yaml_with_envvars(io.StringIO(sample_yaml), {})
