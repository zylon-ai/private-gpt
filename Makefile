# Any args passed to the make script, use with $(call args, default_value)
args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`
AUTO_DISCOVER_ARGS ?=
PROD_DIST_DIR ?= .dist
PROD_PYTHON ?= .venv/bin/python
PROD_BINARY ?= .venv/bin/private-gpt
PROD_ARGS ?= serve
PROD_UV_CACHE_DIR ?= .uv-cache

.PHONY: test test-coverage black ruff format mypy check auto-discover-models run dev-windows dev prod-run api-docs docs ingest wipe celery flower

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	rm -rf local_data/tests/*
	PYTHONPATH=. uv run pytest tests

test-coverage:
	rm -rf local_data/tests/*
	PYTHONPATH=. uv run pytest tests --cov private_gpt --cov-report term --cov-report=html --cov-report xml --junit-xml=tests-results.xml

black:
	uv run black . --check

ruff:
	uv run ruff check private_gpt tests scripts

format:
	uv run black .
	uv run ruff check private_gpt tests scripts --fix

mypy:
	@if ! uv run dmypy status >/dev/null 2>&1; then \
		echo "Starting mypy daemon..."; \
		uv run dmypy start; \
	fi
	uv run dmypy check private_gpt scripts

check:
	make format
	make mypy

auto-discover-models:
	uv run python scripts/auto_discover_models.py $(AUTO_DISCOVER_ARGS)


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
	@mkdir -p local_data
	@find local_data -mindepth 1 ! -name '.gitignore' -exec rm -rf {} +
########################################################################################################################
# Celery
########################################################################################################################

celery:
	PGPT_WORKER_MODE=worker uv run private-gpt worker

flower:
	PGPT_WORKER_MODE=flower uv run private-gpt worker
