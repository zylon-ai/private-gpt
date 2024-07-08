# Any args passed to the make script, use with $(call args, default_value)
args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	PYTHONPATH=. poetry run pytest tests

test-coverage:
	PYTHONPATH=. poetry run pytest tests --cov private_gpt --cov-report term --cov-report=html --cov-report xml --junit-xml=tests-results.xml

black:
	poetry run black . --check

ruff:
	poetry run ruff check private_gpt tests

format:
	poetry run black .
	poetry run ruff check private_gpt tests --fix

mypy:
	poetry run mypy private_gpt

check:
	make format
	make mypy

########################################################################################################################
# Run
########################################################################################################################

run:
	poetry run python -m private_gpt

dev-windows:
	(set PGPT_PROFILES=local & poetry run python -m uvicorn private_gpt.main:app --reload --port 8001)

dev:
	PYTHONUNBUFFERED=1 PGPT_PROFILES=local poetry run python -m uvicorn private_gpt.main:app --reload --port 8001

########################################################################################################################
# Misc
########################################################################################################################

api-docs:
	PGPT_PROFILES=mock poetry run python scripts/extract_openapi.py private_gpt.main:app --out fern/openapi/openapi.json

ingest:
	@poetry run python scripts/ingest_folder.py $(call args)

stats:
	poetry run python scripts/utils.py stats

wipe:
	poetry run python scripts/utils.py wipe

setup:
	poetry run python scripts/setup

list:
	@echo "Available commands:"
	@echo "  test            : Run tests using pytest"
	@echo "  test-coverage   : Run tests with coverage report"
	@echo "  black           : Check code format with black"
	@echo "  ruff            : Check code with ruff"
	@echo "  format          : Format code with black and ruff"
	@echo "  mypy            : Run mypy for type checking"
	@echo "  check           : Run format and mypy commands"
	@echo "  run             : Run the application"
	@echo "  dev-windows     : Run the application in development mode on Windows"
	@echo "  dev             : Run the application in development mode"
	@echo "  api-docs        : Generate API documentation"
	@echo "  ingest          : Ingest data using specified script"
	@echo "  wipe            : Wipe data using specified script"
	@echo "  setup           : Setup the application"
