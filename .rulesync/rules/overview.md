---
root: true
targets: ["*"]
description: "Project overview and general development guidelines"
globs: ["**/*"]
---

## Project Overview

This is a python project that enforces strict typing and linting standards. 
Uses `uv` as the package manager with `hatchling` as the build backend.

## Commands

```bash
make install      # Install dependencies (uv sync)
make build        # Build the project
make typecheck    # Run mypy type checking
make test         # Run pytest
make lint         # Run ruff linter
make deptry       # Check for dependency issues
make pre-commit   # Run all pre-commit hooks
```

Run a single test file:

```bash
uv run pytest tests/test_utils.py
```

Run a single test function:

```bash
uv run pytest tests/test_utils.py::test_add
```

## Style and Type Guide for AI Agents

### Type Annotations Are Mandatory

All function definitions must include complete type annotations. The mypy configuration enforces `disallow_untyped_defs = true`.

```python
# Correct
def process(data: list[str], count: int) -> dict[str, int]:
    ...

# Wrong - will fail type checking
def process(data, count):
    ...
```

### Use Modern Python Typing

Target Python 3.12+. Use built-in generic types, not `typing` module equivalents:

```python
# Correct (Python 3.12+)
def get_items() -> list[str]:
    ...

def get_mapping() -> dict[str, int]:
    ...

# Avoid (legacy style)
from typing import List, Dict
def get_items() -> List[str]:
    ...
```

Use `Optional` from typing or the `X | None` union syntax:

```python
from typing import Optional

def find(name: Optional[str] = None) -> str:
    ...

# Or equivalently:
def find(name: str | None = None) -> str:
    ...
```

### Line Length

Maximum line length is 120 characters (configured in ruff).

### Pre-commit Hooks

The following checks run automatically:

- **On commit**: Type checking (mypy), Linting (ruff)
- **On push**: Build verification, Dependency check (deptry)

### Test Files

Test files are excluded from mypy checking (`exclude = "test_*"` in pyproject.toml), but should still follow typing conventions for consistency.

## Linting Rules (Ruff)

Ruff is configured with the following rule sets:

| Rule Set | Code | Description |
|----------|------|-------------|
| pydocstyle | `D` | Docstring conventions |
| pycodestyle | `E`, `W` | Errors and warnings |
| pyflakes | `F` | Logical errors |
| isort | `I` | Import sorting |
| flake8-bugbear | `B` | Bug detection |
| flake8-comprehensions | `C4` | Comprehension best practices |
| pyupgrade | `UP` | Python version upgrades |
| flake8-simplify | `SIM` | Code simplification |
| flake8-type-checking | `TCH` | Type checking imports |
| flake8-use-pathlib | `PTH` | Pathlib usage |
| mccabe | `C901` | Cyclomatic complexity (max: 10) |
| pep8-naming | `N` | Naming conventions |
| flake8-return | `RET` | Return statement checks |
| tryceratops | `TRY` | Exception handling |
| perflint | `PERF` | Performance checks |
| ruff-specific | `RUF` | Ruff-specific rules |
| eradicate | `ERA` | Commented-out code detection |
| flake8-print | `T20` | Print statement detection |
| pylint refactor | `PLR` | Refactoring suggestions |

### Pylint Limits

- **max-args**: 5 arguments per function
- **max-branches**: 12 branches per function
- **max-statements**: 50 statements per function

## Docstring Conventions

This project uses the **NumPy docstring convention**. All public modules, classes, and functions must have docstrings.

### Test Files Exception

Test files (`tests/**`) are exempt from all docstring rules (`D`).

### NumPy Docstring Format

```python
def calculate_area(width: float, height: float) -> float:
    """
    Calculate the area of a rectangle.

    Parameters
    ----------
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.

    Returns
    -------
    float
        The calculated area.

    Examples
    --------
    >>> calculate_area(3.0, 4.0)
    12.0
    """
    return width * height
```

### Class Docstring Example

```python
class DataProcessor:
    """
    Process and transform data records.

    Attributes
    ----------
    source : str
        The data source identifier.
    batch_size : int
        Number of records to process per batch.

    Methods
    -------
    process(data)
        Process a single data record.
    batch_process(records)
        Process multiple records in batches.
    """
```

