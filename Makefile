# Any args passed to the make script, use with $(call args, default_value)
args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`
AUTO_DISCOVER_ARGS ?=
WORKER_ARGS ?=
PROD_DIST_DIR ?= .dist
PROD_PYTHON ?= .venv/bin/python
PROD_BINARY ?= .venv/bin/private-gpt
PROD_ARGS ?= serve
PROD_UV_CACHE_DIR ?= .uv-cache
TEST_PGPT_HOME ?= $(CURDIR)
TEST_LOCAL_DATA_DIR ?= $(TEST_PGPT_HOME)/local_data/tests
WIPE_PGPT_HOME := $(if $(PGPT_HOME),$(PGPT_HOME),$(HOME)/.local/share/private-gpt)
WIPE_LOCAL_DATA_DIR := $(WIPE_PGPT_HOME)/local_data

.PHONY: test test-coverage format lint typecheck fix check auto-discover-models update-openapi-spec run dev-windows dev prod-run api-docs docs ingest wipe celery flower celery-worker arq-worker chat-worker tools-worker

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	rm -rf "$(TEST_LOCAL_DATA_DIR)"/*
	PGPT_HOME=$(TEST_PGPT_HOME) PYTHONPATH=. uv run pytest tests

test-coverage:
	rm -rf "$(TEST_LOCAL_DATA_DIR)"/*
	PGPT_HOME=$(TEST_PGPT_HOME) PYTHONPATH=. uv run pytest tests --cov private_gpt --cov-report term --cov-report=html --cov-report xml --junit-xml=tests-results.xml

format:
	uv run ruff format . --check

lint:
	uv run ruff check private_gpt tests scripts

fix:
	uv run ruff check private_gpt tests scripts --fix
	uv run ruff format .

typecheck:
	uv run ty check private_gpt scripts

check:
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) typecheck

auto-discover-models:
	uv run python scripts/auto_discover_models.py $(AUTO_DISCOVER_ARGS)

update-openapi-spec:
	uv run python scripts/update_claude_openapi.py


########################################################################################################################
# Run
########################################################################################################################

run:
	uv run private-gpt serve

dev-windows:
	uv run private-gpt serve --reload

dev:
	PYTHONUNBUFFERED=1 uv run private-gpt serve --reload --host 0.0.0.0 --port 8080

prod-run:
	rm -rf $(PROD_DIST_DIR)
	mkdir -p $(PROD_DIST_DIR)
	UV_CACHE_DIR=$(PROD_UV_CACHE_DIR) uv build --out-dir $(PROD_DIST_DIR)
	@if [ ! -x "$(PROD_PYTHON)" ]; then \
		echo "Python interpreter not found at $(PROD_PYTHON). Create the project venv first."; \
		exit 1; \
	fi
	@wheel="$$(ls -t $(PROD_DIST_DIR)/private_gpt-*.whl 2>/dev/null | head -n 1)"; \
	if [ -z "$$wheel" ]; then \
		echo "No built wheel found in $(PROD_DIST_DIR)."; \
		exit 1; \
	fi; \
	UV_CACHE_DIR=$(PROD_UV_CACHE_DIR) uv pip install --python $(PROD_PYTHON) --force-reinstall --no-deps "$$wheel"; \
	$(PROD_BINARY) $(PROD_ARGS)

########################################################################################################################
# Misc
########################################################################################################################

api-docs:
	PGPT_PROFILES=mock uv run python scripts/extract_openapi.py private_gpt.main:app --out fern/openapi/openapi.json

docs:
	cd fern && npx fern-api@latest docs check && npx fern-api@latest docs dev

ingest:
	@uv run python scripts/ingest_folder.py $(call args)

wipe:
	@mkdir -p "$(WIPE_LOCAL_DATA_DIR)"
	@find "$(WIPE_LOCAL_DATA_DIR)" -mindepth 1 ! -name '.gitignore' -exec rm -rf {} +
########################################################################################################################
# Celery
########################################################################################################################

celery:
	PGPT_WORKER_MODE=celery ./scripts/worker_entrypoint $(WORKER_ARGS)

flower:
	PGPT_WORKER_MODE=flower ./scripts/worker_entrypoint $(WORKER_ARGS)

celery-worker:
	PGPT_WORKER_MODE=celery ./scripts/worker_entrypoint $(WORKER_ARGS)

arq-worker:
	PGPT_WORKER_MODE=arq ./scripts/worker_entrypoint $(WORKER_ARGS)

chat-worker:
	PGPT_WORKER_MODE=arq \
	PGPT_WORKER_APP_MODULE=private_gpt \
	PGPT_ARQ_QUEUE=chat \
	PGPT_ARQ_TASK_PACKAGES=private_gpt.arq.tasks.chat \
	PGPT_STATEFUL_WORKER_TYPE=chat \
	PGPT_WORKER_WARM_PROFILE=chat \
	./scripts/worker_entrypoint $(WORKER_ARGS)

tools-worker:
	PGPT_WORKER_MODE=celery \
	PGPT_WORKER_APP_MODULE=private_gpt \
	PGPT_CELERY_QUEUES=tools \
	PGPT_CELERY_TASK_PACKAGES=private_gpt.celery.tasks.tools \
	PGPT_STATEFUL_WORKER_TYPE=tools \
	PGPT_WORKER_WARM_PROFILE=tools \
	./scripts/worker_entrypoint $(WORKER_ARGS)
