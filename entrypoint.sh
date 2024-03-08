#!/bin/sh

## Choose the model, tokenizer and prompt style
export PGPT_HF_REPO_ID="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
export PGPT_HF_MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
export PGPT_TOKENIZER="mistralai/Mistral-7B-Instruct-v0.2"
export PGPT_PROMPT_STYLE="mistral"

## Optionally, choose a different embedding model
# export PGPT_EMBEDDING_HF_MODEL_NAME="BAAI/bge-small-en-v1.5"

## Download the embedding and model files
echo "Running setup script"
poetry run python scripts/setup

## Execute the main container command
exec "$@"