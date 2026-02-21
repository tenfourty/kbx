# Contributing to kbx

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/tenfourty/kbx.git
cd kbx
uv sync --all-extras
uv run pre-commit install
```

## Workflow

1. Create a branch for your change
2. Write tests first (TDD)
3. Implement the change
4. Run the full check suite:
   ```bash
   uv run pytest -x -q --cov
   uv run mypy src/
   uv run ruff check src/
   ```
5. Open a pull request

## Code Style

- Python 3.10+ (no walrus operators in hot paths)
- Strict mypy, Pydantic v2 (`strict=True`)
- Line length 100 (ruff enforced)
- Coverage minimum 90%

## Architecture

See `docs/architecture.md` for the system design.
