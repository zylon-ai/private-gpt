#!/usr/bin/env bash

# --------------------------------------------------------------
PIP_LOCK="pip.lock"
if [ ! -f "$PIP_LOCK" ];then
  pip install --root-user-action=ignore -r requirements.txt
  touch "$PIP_LOCK"
fi

# --------------------------------------------------------------
DIR="/models"
GGML_GPT="https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
if [ -z "$(ls -A $DIR)" ]; then
  echo "Downloading the model $GGML_GPT."
  wget $GGML_GPT -P /models
fi

# --------------------------------------------------------------
SENTENCE_TRANSFORMERS="sentence_transformers.lock"
if [ ! -f "$SENTENCE_TRANSFORMERS" ]; then
  pip install sentence_transformers
  touch "$SENTENCE_TRANSFORMERS"
fi

# --------------------------------------------------------------
POETRY_LOCK="poetry.lock"
if [ ! -f "$POETRY_LOCK" ]; then
  pip install poetry
  #poetry env use python3.11
  #poetry config installer.max-workers 10
  poetry --no-interaction --no-ansi -vvv --no-root install
  touch "$POETRY_LOCK"
fi

# --------------------------------------------------------------
POETRY_SHELL_LOCK="poetryShell.lock"
if [ ! -f "$POETRY_SHELL_LOCK" ]; then
  poetry shell
  touch $POETRY_SHELL_LOCK
fi

# --------------------------------------------------------------
INGEST_LOCK="ingest.lock"
if [ ! -f "$INGEST_LOCK" ]; then
  python ingest.py
  touch $INGEST_LOCK
fi

echo "--------------------------------------------------------------"
echo "To ingest data: 'python ingest'"
echo "To ask question: 'python privateGPT.py'"
echo "--------------------------------------------------------------"

exec "$@"
