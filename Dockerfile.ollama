FROM python:3.11.6-slim-bookworm AS base

# Install poetry
RUN pip install pipx
RUN python3 -m pipx ensurepath
RUN pipx install poetry==1.8.3
ENV PATH="/root/.local/bin:$PATH"
ENV PATH=".venv/bin/:$PATH"

# https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

FROM base AS dependencies
WORKDIR /home/worker/app
COPY pyproject.toml poetry.lock ./

ARG POETRY_EXTRAS="ui vector-stores-qdrant llms-ollama embeddings-ollama"
RUN poetry install --no-root --extras "${POETRY_EXTRAS}"

FROM base AS app
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV APP_ENV=prod
ENV PYTHONPATH="$PYTHONPATH:/home/worker/app/private_gpt/"
EXPOSE 8080

# Prepare a non-root user
# More info about how to configure UIDs and GIDs in Docker:
# https://github.com/systemd/systemd/blob/main/docs/UIDS-GIDS.md

# Define the User ID (UID) for the non-root user
# UID 100 is chosen to avoid conflicts with existing system users
ARG UID=100

# Define the Group ID (GID) for the non-root user
# GID 65534 is often used for the 'nogroup' or 'nobody' group
ARG GID=65534

RUN adduser --system --gid ${GID} --uid ${UID} --home /home/worker worker
WORKDIR /home/worker/app

RUN chown worker /home/worker/app
RUN mkdir local_data && chown worker local_data
RUN mkdir models && chown worker models
COPY --chown=worker --from=dependencies /home/worker/app/.venv/ .venv
COPY --chown=worker private_gpt/ private_gpt
COPY --chown=worker *.yaml .
COPY --chown=worker scripts/ scripts

USER worker
ENTRYPOINT python -m private_gpt
