# Makefile for the moe-routing package.
# Targets: test, lint, format, bench, docs, clean.

.PHONY: test lint format bench docs clean help

PYTHON ?= python3

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

test:  ## Run the full test suite with coverage.
	$(PYTHON) -m pytest

lint:  ## Run ruff and mypy static checks.
	$(PYTHON) -m ruff check moe tests
	$(PYTHON) -m mypy moe

format:  ## Auto-format the code with ruff.
	$(PYTHON) -m ruff format moe tests
	$(PYTHON) -m ruff check --fix moe tests

bench:  ## Benchmark a forward pass: MoE vs equivalent dense FFN.
	$(PYTHON) -m moe.bench

docs:  ## Generate HTML API docs from docstrings via pdoc.
	$(PYTHON) -m pdoc moe -o docs/api

clean:  ## Remove caches and build artefacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
