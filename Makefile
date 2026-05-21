.PHONY: install install-dev install-integration lint format type test test-integration clean build

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[sql,test]"

install-integration:
	uv pip install -e ".[sql,test,test-integration]"

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

type:
	uv run mypy langchain_chdb

# Default unit + contract test target. Skips integration tests requiring HF / network.
test:
	uv run pytest -v -m "not integration"

# Nightly / local target. Requires the [test-integration] extra (HF embedder).
test-integration:
	uv run pytest -v

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

build:
	rm -rf dist
	uv build
	uv run --with twine twine check dist/*
