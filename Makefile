UV ?= uv
PYTHON ?= 3.12
RUN = $(UV) run --no-sync

.PHONY: setup check lint format typecheck test docs build versions

setup:
	$(UV) sync --frozen --all-packages --group dev --python $(PYTHON) \
		--extra jax --extra gnn --extra mamba --extra cp \
		--extra report --extra analysis --extra experiments

check: versions lint typecheck test docs build

versions:
	$(RUN) python scripts/version.py --check

lint:
	$(RUN) ruff check packages scripts
	$(RUN) ruff format --check packages scripts

format:
	$(RUN) ruff check --fix packages scripts
	$(RUN) ruff format packages scripts

typecheck:
	$(RUN) pyright packages

test:
	$(RUN) pytest packages -m "not gpu and not slow"

docs:
	$(RUN) mkdocs build --strict

build:
	$(UV) build --all-packages --out-dir dist
