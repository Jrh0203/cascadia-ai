.PHONY: check test format format-check lint cluster-test v3-test v3-schema real-root-test clean-local

HOST_TOOL_PATH := /opt/homebrew/bin:/usr/local/bin:/opt/homebrew/opt/rustup/bin:/usr/local/opt/rustup/bin:$(PATH)
UV := $(or $(firstword $(wildcard /opt/homebrew/bin/uv /usr/local/bin/uv)),$(shell command -v uv 2>/dev/null),uv)
CARGO_BIN := $(or $(firstword $(wildcard /opt/homebrew/opt/rustup/bin/cargo /usr/local/opt/rustup/bin/cargo)),$(shell command -v cargo 2>/dev/null),cargo)
CARGO := /usr/bin/env PATH=$(HOST_TOOL_PATH) $(CARGO_BIN)
PYTHON ?= python3

PYTHONPATH_ACTIVE := PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:cascadiav3/src:tools
RUFF_PATHS := python/cascadia_cluster tests/cluster_unit tools/cluster_*.py tools/r2_map_bacalhau_gate.py

check:
	$(CARGO) check --workspace
	$(CARGO) check --manifest-path cascadiav3/real-root-exporter/Cargo.toml
	$(PYTHONPATH_ACTIVE) $(PYTHON) -m cascadiav3.validate_schema_registry --include-legacy --include-expert

test: cluster-test v3-test real-root-test
	$(CARGO) test --workspace

cluster-test:
	$(PYTHONPATH_ACTIVE) $(UV) run pytest -q tests/cluster_unit tools/test_cluster_*.py

v3-test:
	$(PYTHONPATH_ACTIVE) $(PYTHON) -m unittest discover -s cascadiav3/tests -v

v3-schema:
	$(PYTHONPATH_ACTIVE) $(PYTHON) -m cascadiav3.validate_schema_registry --include-legacy --include-expert

real-root-test:
	$(CARGO) test --manifest-path cascadiav3/real-root-exporter/Cargo.toml

format:
	$(CARGO) fmt --all
	$(UV) run ruff format $(RUFF_PATHS)

format-check:
	$(CARGO) fmt --all -- --check
	$(UV) run ruff format --check $(RUFF_PATHS)

lint:
	$(CARGO) clippy --workspace --all-targets --no-deps -- -D warnings
	$(UV) run ruff check --no-cache $(RUFF_PATHS)

clean-local:
	$(CARGO) clean
	rm -rf cascadiav3/real-root-exporter/target .venv node_modules
