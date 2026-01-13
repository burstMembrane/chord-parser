# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description

A Python library for parsing and converting between chord representations (e.g., Harte notation via `harte-library` and simple chord notation via `pychord`).

## Commands

```bash
make install      # Install dependencies (uv sync)
make build        # Build the project
make typecheck    # Run mypy type checking
make test         # Run pytest
make lint         # Run ruff linter
make format       # Format code with ruff
make deptry       # Check for dependency issues
make bandit       # Security scan
make audit        # Vulnerability scan (pip-audit)
make pre-commit   # Run all pre-commit hooks
make pre-push     # Full validation (lint, typecheck, test, bandit, audit, deptry)
```

Run a single test:

```bash
uv run pytest tests/test_file.py::test_function
```

## Code Standards

- **Python 3.12+** with strict typing enforced by mypy
- **Type annotations required** on all function definitions
- Use modern Python typing (`list[str]`, `dict[str, int]`, `X | None`) not legacy `typing.List`, `typing.Dict`
- **NumPy docstring convention** for all public modules, classes, and functions
- Max line length: 120 characters
- Max function args: 5, branches: 12, statements: 50, complexity: 10

## Project Structure

- Source code: `chord_parser/`
- Tests: `tests/` (exempt from docstring rules)
- Package manager: `uv` with `hatchling` build backend
- CLI entry point: `test-cli` command (defined in pyproject.toml)

## Key Dependencies

- `harte-library`: Harte chord notation parsing
- `pychord`: Simple chord notation parsing

## Pre-commit Hooks

On commit: typecheck, test, lint, format-check, bandit
On push: build, deptry, pip-audit
