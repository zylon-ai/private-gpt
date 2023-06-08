.PHONY: dev
dev:
	python3.11 -m venv .venv --upgrade-deps
	.venv/bin/pip3 install -r requirements.txt
	.venv/bin/pip3 install -r infra-requirements.txt
	.venv/bin/pre-commit install

.PHONY: fix-all
fix-all:
	.venv/bin/pre-commit run --all-files

.PHONY: lint
lint:
	.venv/bin/ruff .

.PHONY: format
format:
	.venv/bin/black --check .

.PHONY: format-fix
format-fix:
	.venv/bin/ruff check . --fix --fixable I
	.venv/bin/black .

.PHONY: lint-fix
lint-fix:
	.venv/bin/ruff check . --fix