FROM python:3.11.6-slim-bullseye as builder

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

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

FROM builder as prod-dependencies
WORKDIR /build
COPY pyproject.toml poetry.lock ./

# Build will fail until this is fixed:
# https://github.com/grantjenks/py-tree-sitter-languages/pull/27
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN poetry install --no-root --no-dev

FROM prod-dependencies as dev-dependencies
# Install dev dependencies too
WORKDIR /build
RUN poetry install

FROM python:3-slim-bullseye as base-app

ARG DEPENDENCIES=prod-dependencies
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

# Prepare a non-root user
RUN adduser --system worker
WORKDIR /home/worker/app

# Copy everything, including the virtual environment
COPY --chown=worker . private_gpt

USER worker
ENTRYPOINT .venv/bin/python -m private_gpt

FROM base-app as prod-app
COPY --chown=worker --from=prod-dependencies /build /app

FROM base-app as dev-app
COPY --chown=worker --from=dev-dependencies /build /app