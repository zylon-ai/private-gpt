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

wipe:
	poetry run python scripts/utils.py wipe

setup:
	poetry run python scripts/setup
