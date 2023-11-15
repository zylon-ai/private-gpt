#!/usr/bin/env python3
import os
import argparse

from huggingface_hub import hf_hub_download, snapshot_download

from private_gpt.paths import models_path, models_cache_path
from private_gpt.settings.settings import settings

resume_download = True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Setup: Download models from huggingface')
    parser.add_argument('--resume', default=True, action=argparse.BooleanOptionalAction, help='Enable/Disable resume_download options to restart the download progress interrupted')
    args = parser.parse_args()
    resume_download = args.resume

os.makedirs(models_path, exist_ok=True)
embedding_path = models_path / "embedding"

print(f"Downloading embedding {settings().local.embedding_hf_model_name}")
snapshot_download(
    repo_id=settings().local.embedding_hf_model_name,
    cache_dir=models_cache_path,
    local_dir=embedding_path,
)
print("Embedding model downloaded!")
print("Downloading models for local execution...")

# Download LLM and create a symlink to the model file
hf_hub_download(
    repo_id=settings().local.llm_hf_repo_id,
    filename=settings().local.llm_hf_model_file,
    cache_dir=models_cache_path,
    local_dir=models_path,
    resume_download=resume_download,
)

print("LLM model downloaded!")
print("Setup done")
