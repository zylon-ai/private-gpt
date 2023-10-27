import io
import os

import pytest

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
