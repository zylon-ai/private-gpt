### IMPORTANT, THIS IMAGE CAN ONLY BE RUN IN LINUX DOCKER
### You will run into a segfault in mac
# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as base
ENV PYTHON_VERSION=3.11.6


ENV TZ_SELECTION=2
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        && \
    rm -rf /var/lib/apt/lists/*

ENV HOME="/root"
WORKDIR ${HOME}
RUN apt-get install -y git
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

# Verify Python installation
RUN python --version

# Install poetry
RUN pip install pipx
RUN python3.11 -m pipx ensurepath
RUN pipx install poetry
ENV PATH="/root/.local/bin:$PATH"

# Dependencies to build llama-cpp
RUN apt update && apt install -y \
  libopenblas-dev\
  ninja-build\
  build-essential\
  pkg-config\
  wget

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

FROM base as dependencies
WORKDIR /saimon
COPY pyproject.toml poetry.lock ./

RUN poetry install --with local
RUN poetry install --with ui
# GPU SUPPORT FOR LLAMA
RUN CMAKE_ARGS='-DLLAMA_CUBLAS=on' poetry run pip install --force-reinstall --no-cache-dir llama-cpp-python

FROM base as app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /saimon

COPY . .
RUN mkdir local_data
RUN mkdir models
COPY --from=dependencies /saimon/.venv/ .venv

RUN poetry run python setup

ENV PGPT_PROFILES=local
ENTRYPOINT .venv/bin/python -u -m private_gpt