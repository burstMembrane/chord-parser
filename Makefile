.PHONY: install venv build typecheck test lint format format-check deptry bandit audit pre-commit pre-push rulesync clean bootstrap
install:
	uv sync 
venv:
	uv venv
build:
	uv build
typecheck:
	uv run mypy chord_parser
test: 
	uv run pytest tests 
lint:
	uv run ruff check chord_parser tests
format:
	uv run ruff format chord_parser tests
format-check:
	uv run ruff format --check chord_parser tests
deptry:
	uv run deptry chord_parser

pre-commit:
	uv run pre-commit run --all-files
pre-push: lint typecheck test bandit audit deptry
rulesync:
	npx rulesync generate
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
bootstrap:
	echo "3.12" >> .python-version
	uv sync
	git init
	uv run pre-commit install
	uv run pytest tests
	uv run ruff check chord_parser tests
	uv run test-cli --name "chord_parser"
