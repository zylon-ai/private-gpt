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

run:
	poetry run python -m private_gpt

dev:
	PYTHONUNBUFFERED=1 PGPT_PROFILES=local poetry run python -m uvicorn private_gpt.main:app --reload --port 8001

install-mac:
	CMAKE_ARGS="-DLLAMA_METAL=on" poetry install

install-windows:
