ARG EXTRAS="core"

FROM python:3.11.10-slim-bookworm AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV PATH="/usr/local/bin:${PATH}"
ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

FROM base AS os-deps
ARG EXTRAS
RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eu; \
    ENABLE_DOCUMENTS=0; \
    ENABLE_MEDIA=0; \
    ENABLE_WEB_SCRAPING=0; \
    ENABLE_DATABASE=0; \
    for extra in $EXTRAS; do \
      case "$extra" in \
        ingest|ingest-core|ingest-documents|ingest-markitdown|ingest-markup|core) ENABLE_DOCUMENTS=1 ;; \
        media|media-core|media-image|media-audio|media-video) ENABLE_MEDIA=1 ;; \
        tools|tool-web-scraping|core) ENABLE_WEB_SCRAPING=1 ;; \
        database|database-core|database-postgres|database-mysql|database-mssql|database-db2|tool-database) ENABLE_DATABASE=1 ;; \
      esac; \
    done; \
    apt update; \
    apt install -yqq --no-install-recommends \
      curl \
      ca-certificates; \
    if [ "$ENABLE_DOCUMENTS" = "1" ]; then \
      apt install -yqq --no-install-recommends \
        libmagic1 \
        libxml2 \
        libxslt1.1 \
        poppler-utils \
        libreoffice \
        pandoc \
        ghostscript; \
    fi; \
    if [ "$ENABLE_MEDIA" = "1" ]; then \
      apt install -yqq --no-install-recommends \
        ffmpeg \
        libavcodec-extra; \
    fi; \
    if [ "$ENABLE_WEB_SCRAPING" = "1" ]; then \
      apt install -yqq --no-install-recommends \
        libglib2.0-0 \
        libnss3 \
        libnspr4 \
        libdbus-1-3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libatspi2.0-0 \
        libcups2 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libxcb1 \
        libxkbcommon0 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
        libxfont2 \
        x11-xkb-utils \
        xserver-common; \
    fi; \
    if [ "$ENABLE_DATABASE" = "1" ]; then \
      apt install -yqq --no-install-recommends libmariadb3; \
      curl -sSL -O https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb; \
      dpkg -i packages-microsoft-prod.deb; \
      rm packages-microsoft-prod.deb; \
      apt update; \
      ACCEPT_EULA=Y apt install -yqq --no-install-recommends msodbcsql18; \
    fi; \
    rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
WORKDIR /home/worker/app

ARG EXTRAS

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    set -eu; \
    apt update; \
    apt install -yqq --no-install-recommends \
      build-essential \
      pkg-config; \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    extra_flags=""; \
    for extra in $EXTRAS; do \
      extra_flags="$extra_flags --extra $extra"; \
    done; \
    uv sync --no-dev --frozen --no-install-project $extra_flags

# remove hf_xet, it does not support custom CA which
# makes them fail in some corporate environments.
# The fallback is hf-transfer, which uses HTTPS properly (with certificate validation)
RUN uv pip uninstall hf-xet

FROM base AS build
WORKDIR /home/worker/app
COPY private_gpt/ private_gpt
COPY ui/ ui

# New download stage for caching model files
FROM base AS downloads
WORKDIR /home/worker/app
ARG EXTRAS
ARG PGPT_DOWNLOAD_NLTK
ARG PGPT_DOWNLOAD_TIKTOKEN_CACHE
ARG PGPT_DOWNLOAD_TIKTOKEN_ENCODINGS
ARG PGPT_DOWNLOAD_PLAYWRIGHT

COPY pyproject.toml uv.lock ./

ENV HF_HOME=local_data
ENV PLAYWRIGHT_BROWSERS_PATH=/home/worker/app/.local-browsers
ENV TIKTOKEN_CACHE_DIR=/home/worker/app/tiktoken_cache
ENV TIKTOKEN_ENCODINGS_BASE=/home/worker/app/encodings
ENV TIKTOKEN_RS_CACHE_DIR=/home/worker/app/encodings

RUN mkdir -p models/nltk_cache .local-browsers tiktoken_cache encodings local_data

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    has_web_scraping=0; \
    for extra in $EXTRAS; do \
      if [ "$extra" = "tools" ] || [ "$extra" = "tool-web-scraping" ] || [ "$extra" = "core" ]; then \
        has_web_scraping=1; \
        break; \
      fi; \
    done; \
    resolved_download_nltk="${PGPT_DOWNLOAD_NLTK:-1}"; \
    resolved_download_tiktoken_cache="${PGPT_DOWNLOAD_TIKTOKEN_CACHE:-1}"; \
    resolved_download_tiktoken_encodings="${PGPT_DOWNLOAD_TIKTOKEN_ENCODINGS:-1}"; \
    resolved_download_playwright="${PGPT_DOWNLOAD_PLAYWRIGHT:-$has_web_scraping}"; \
    requested_packages=""; \
    if [ "$resolved_download_nltk" = "1" ]; then \
      requested_packages="$requested_packages nltk"; \
    fi; \
    if [ "$resolved_download_tiktoken_cache" = "1" ] || [ "$resolved_download_tiktoken_encodings" = "1" ]; then \
      requested_packages="$requested_packages tiktoken"; \
    fi; \
    if [ "$resolved_download_playwright" = "1" ]; then \
      requested_packages="$requested_packages playwright"; \
    fi; \
    if [ -n "$requested_packages" ]; then \
      export REQUESTED_PACKAGES="$requested_packages"; \
      locked_packages="$(python3 -c 'import os, sys, tomllib; requested = [pkg for pkg in os.environ.get("REQUESTED_PACKAGES", "").split() if pkg]; lock = tomllib.load(open("uv.lock", "rb")); versions = {pkg["name"]: pkg["version"] for pkg in lock.get("package", []) if pkg.get("name") in requested and "version" in pkg}; missing = [pkg for pkg in requested if pkg not in versions]; sys.exit("Missing locked version(s) in uv.lock for: " + ", ".join(missing)) if missing else print(" ".join(f"{pkg}=={versions[pkg]}" for pkg in requested))')"; \
      uv pip install --system $locked_packages; \
    fi; \
    if [ "$resolved_download_nltk" = "1" ]; then \
      python3 -c "import nltk, zipfile; from pathlib import Path; download_dir=Path('models/nltk_cache'); packages=('punkt_tab', 'punkt', 'averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'stopwords', 'wordnet'); [nltk.download(p, download_dir=str(download_dir), raise_on_error=True) for p in packages]; [(lambda path: (lambda zf: (zf.extractall(path.parent), zf.close()))(zipfile.ZipFile(path)))(path) for path in download_dir.rglob('*.zip')]"; \
    fi; \
    if [ "$resolved_download_tiktoken_cache" = "1" ]; then \
      python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"; \
    fi; \
    if [ "$resolved_download_tiktoken_encodings" = "1" ]; then \
      python3 -c "import urllib.request; from pathlib import Path; d = Path('encodings'); d.mkdir(exist_ok=True); [urllib.request.urlretrieve(f'https://openaipublic.blob.core.windows.net/encodings/{name}', d / name) for name in ('o200k_base.tiktoken', 'cl100k_base.tiktoken')]"; \
    fi; \
    if [ "$resolved_download_playwright" = "1" ]; then \
      playwright install chromium; \
    fi

FROM os-deps AS final

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

RUN adduser worker
WORKDIR /home/worker/app
RUN mkdir -p /home/worker/app /home/worker/app/local_data /home/worker/app/models /home/worker/app/scripts /home/worker/app/ui \
    && chown -R worker /home/worker/app

# Copy virtual environment
COPY --chown=worker --from=dependencies /home/worker/app/pyproject.toml /home/worker/app/pyproject.toml
COPY --chown=worker --from=dependencies /home/worker/app/uv.lock /home/worker/app/uv.lock
COPY --chown=worker --from=dependencies /home/worker/app/.venv/ /home/worker/app/.venv/

# Copy application code
COPY --chown=worker --from=build /home/worker/app/ /home/worker/app/

# Copy downloaded model files
COPY --chown=worker --from=downloads /home/worker/app/models/nltk_cache/ models/nltk_cache/
COPY --chown=worker --from=downloads /home/worker/app/.local-browsers/ .local-browsers/
COPY --chown=worker --from=downloads /home/worker/app/tiktoken_cache/ tiktoken_cache/
COPY --chown=worker --from=downloads /home/worker/app/encodings/ encodings/

# Copy additional files
COPY --chown=worker scripts/worker_entrypoint scripts/worker_entrypoint
RUN chmod +x scripts/worker_entrypoint

COPY --chown=worker version.txt version.txt
COPY --chown=worker settings.yaml settings.yaml

RUN uv pip install --python /home/worker/app/.venv/bin/python --no-deps .

ENV PATH="/home/worker/app/.venv/bin:/usr/local/bin:${PATH}"
ENV HF_HOME=local_data
ENV PYTHONPATH="$PYTHONPATH:/private_gpt/"
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

ENV PLAYWRIGHT_BROWSERS_PATH=/home/worker/app/.local-browsers
ENV TIKTOKEN_ENCODINGS_BASE=/home/worker/app/encodings
ENV TIKTOKEN_RS_CACHE_DIR=/home/worker/app/encodings

ENTRYPOINT ["private-gpt"]
CMD ["serve"]
