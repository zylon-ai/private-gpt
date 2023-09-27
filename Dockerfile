FROM python:3-slim-bullseye as builder

# Install poetry
RUN pip install pipx
RUN python3 -m pipx ensurepath
RUN pipx install poetry
ENV PATH="/root/.local/bin:$PATH"

# Dependencies to build llama-cpp and wget
RUN apt update && apt install -y \
  libopenblas-dev\
  ninja-build\
  build-essential\
  pkg-config\
  wget

FROM builder as build

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN poetry install --no-dev

FROM python:3-slim-bullseye
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

# Prepare a non-root user
RUN adduser --system worker
WORKDIR /home/worker/app

# Copy everything, including the virtual environment
COPY --chown=worker --from=build /app/.venv .venv
COPY --chown=worker private_gpt private_gpt

USER worker
ENTRYPOINT .venv/bin/python -m private_gpt