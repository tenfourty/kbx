.PHONY: ci lint typecheck security test

# Mirror the exact GitHub CI pipeline locally.
# Usage: make ci
ci: lint typecheck security test
	@echo "✅ All CI checks passed."

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/

security:
	uv run bandit -c pyproject.toml -r src/

test:
	uv run pytest -x -q --cov --cov-fail-under=90 -m "not slow"
