#!/bin/bash
set -xe;

# chmod +x run.sh
# run with bash

# # Check if the script is run with sudo privileges
# if [ "$EUID" -ne 0 ]; then
#   echo "Please run this script with sudo or as root."
#   exit 1
# fi

# Update the package list
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Remove unnecessary packages
sudo apt autoremove -y

# Clean up cached package files
sudo apt clean

echo "System update complete."

DEBIAN_FRONTEND=noninteractive sudo apt-get install --yes --quiet --no-install-recommends \
    libopenblas-dev\
    ninja-build\
    build-essential\
    pkg-config\
    wget \
    curl \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3.11-lib2to3 \
    python3.11-gdbm \
    python3.11-tk \
    python3-poetry \
    pip \
    git

curl -sSL https://install.python-poetry.org | python3 -
export POETRY_HOME=/opt/poetry
export POETRY_VIRTUALENVS_IN_PROJECT=true
export TERM=xterm-256color 
export CMAKE_ARGS='-DLLAMA_CUBLAS=on'


python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade wheel pip poetry 
pip install --upgrade ffmpy llama-cpp-python

poetry install --extras "ui llms-llama-cpp embeddings-huggingface vector-stores-qdrant"
poetry run pip install huggingface_hub
poetry run python scripts/setup
poetry run python scripts/setup

export PORT=8080
python3.11 -m private_gpt