.PHONY: verify

TRANSACTION := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOURCE_ARCHIVE := $(TRANSACTION)/source.tar
SOURCE_MANIFEST := $(TRANSACTION)/source-manifest.json
ARCHIVE_VERIFIER := $(TRANSACTION)/archive-verify.py
WORKSPACE := $(TMPDIR)/r2-map-python-boundary
PYTEST_BASETEMP := $(TMPDIR)/r2-map-python-boundary-pytest
TOOLCHAIN := /Users/john2/cascadia-bench/r2-map-v1/cache/runs/run-rust-verify-4dd80652629b716d-v6/rustup/toolchains/stable-aarch64-apple-darwin/bin
CARGO_TARGET_DIR ?= $(TMPDIR)/r2-map-python-boundary-cargo-target
UV := UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$(UV_PROJECT_ENVIRONMENT)-python /opt/homebrew/bin/uv
GATE_PYTHONPATH := $(WORKSPACE)/python:$(WORKSPACE)
RUFF_TARGETS := \
	python/cascadia_mlx/r2_map_*.py \
	python/cascadia_mlx/r2_sparse_mlx_model.py \
	python/tests/test_r2_map_*.py \
	python/tests/test_r2_sparse_mlx_model.py \
	python/tests/test_d6_contract.py \
	tools/r2_map_*.py \
	tools/test_r2_map_*.py
PYTEST_TARGETS := \
	python/tests/test_r2_map_*.py \
	python/tests/test_r2_sparse_mlx_model.py \
	python/tests/test_d6_contract.py \
	tools/test_r2_map_*.py

verify:
	test ! -e "$(WORKSPACE)"
	PYTHONDONTWRITEBYTECODE=1 COPYFILE_DISABLE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" verify \
		--manifest "$(SOURCE_MANIFEST)" --archive "$(SOURCE_ARCHIVE)"
	mkdir -m 0700 "$(WORKSPACE)"
	umask 077 && COPYFILE_DISABLE=1 /usr/bin/tar -xf "$(SOURCE_ARCHIVE)" -C "$(WORKSPACE)"
	PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" verify-tree \
		--manifest "$(SOURCE_MANIFEST)" --repository "$(WORKSPACE)"
	cd "$(WORKSPACE)" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
		tools/r2_map_w0_w5_source_manifest.py --repository . verify "$(SOURCE_MANIFEST)"
	cd "$(WORKSPACE)" && PYTHONPATH="$(GATE_PYTHONPATH)" $(UV) run --frozen ruff check --no-cache $(RUFF_TARGETS)
	test ! -e "$(PYTEST_BASETEMP)" && test ! -L "$(PYTEST_BASETEMP)"
	@status=0; \
		cleanup_pytest_temp() { \
			PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" \
				cleanup-pytest; \
		}; \
		cleanup_or_fail() { cleanup_pytest_temp || exit 70; }; \
		trap cleanup_or_fail 0; \
		trap 'cleanup_or_fail; exit 129' 1; \
		trap 'cleanup_or_fail; exit 130' 2; \
		trap 'cleanup_or_fail; exit 143' 15; \
		cd "$(WORKSPACE)" && \
		PATH="$(TOOLCHAIN):$$PATH" \
		CARGO_TARGET_DIR="$(CARGO_TARGET_DIR)" \
		CARGO_TERM_COLOR=never \
		PYTEST_ADDOPTS= \
		PYTHONPATH="$(GATE_PYTHONPATH)" \
		$(UV) run --frozen pytest -q -p no:cacheprovider \
			--basetemp "$(PYTEST_BASETEMP)" \
			-o tmp_path_retention_policy=none $(PYTEST_TARGETS) || status=$$?; \
		trap - 0 1 2 15; \
		cleanup_or_fail; \
		test ! -e "$(PYTEST_BASETEMP)" && test ! -L "$(PYTEST_BASETEMP)" || exit 71; \
		test -z "$$(/usr/bin/find "$(TMPDIR)" -type l -print -quit)" || exit 72; \
		exit $$status
	cd "$(WORKSPACE)" && PYTHONPATH="$(GATE_PYTHONPATH)" $(UV) run --frozen python tools/r2_map_market_protocol_fixture.py \
		--check tests/fixtures/r2_map/public-market-decision-protocol-v3.json
	cd "$(WORKSPACE)" && PYTHONPATH="$(GATE_PYTHONPATH)" $(UV) run --frozen python tools/r2_map_reference_panels.py \
		--revision v1.1 verify docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json
