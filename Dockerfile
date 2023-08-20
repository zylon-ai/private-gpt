FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/sedzisz/privategptid

COPY *.py /
COPY example.env .env
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY entrypoint.sh entrypoint.sh
COPY source_documents/ source_documents

# Installing base dependecies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install build-essential gcc wget curl -y \
    && apt-get clean

# Update pip
RUN pip install --root-user-action=ignore --upgrade pip

ENTRYPOINT ["/entrypoint.sh"]
