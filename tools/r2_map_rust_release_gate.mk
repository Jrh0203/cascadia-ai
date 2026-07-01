.PHONY: prepare test clippy fmt release
.NOTPARALLEL:

TRANSACTION := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOURCE_ARCHIVE := $(TRANSACTION)/source.tar
SOURCE_MANIFEST := $(TRANSACTION)/source-manifest.json
ARCHIVE_VERIFIER := $(TRANSACTION)/archive-verify.py
WORKSPACE := $(TMPDIR)/r2-map-rust-workspace
TOOLCHAIN := /Users/john2/cascadia-bench/r2-map-v1/cache/runs/run-rust-verify-4dd80652629b716d-v6/rustup/toolchains/stable-aarch64-apple-darwin/bin
PACKAGES := -p cascadia-game -p cascadia-data -p cascadia-r2 -p cascadia-model -p cascadia-search -p cascadia-eval -p cascadia-cli-v2
CARGO := /usr/bin/env PATH="$(TOOLCHAIN):$$PATH" CARGO_TERM_COLOR=never cargo

prepare:
	test ! -e "$(WORKSPACE)"
	PYTHONDONTWRITEBYTECODE=1 COPYFILE_DISABLE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" verify \
		--manifest "$(SOURCE_MANIFEST)" --archive "$(SOURCE_ARCHIVE)"
	mkdir -m 0700 "$(WORKSPACE)"
	umask 077 && COPYFILE_DISABLE=1 /usr/bin/tar -xf "$(SOURCE_ARCHIVE)" -C "$(WORKSPACE)"
	PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" verify-tree \
		--manifest "$(SOURCE_MANIFEST)" --repository "$(WORKSPACE)"
	cd "$(WORKSPACE)" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
		tools/r2_map_w0_w5_source_manifest.py --repository . verify "$(SOURCE_MANIFEST)"

test: prepare
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-search \
		r2_map_runner::tests::heterogeneous_four_model_game_completes_with_exact_focal_exploration_trace -- --exact --ignored
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-game
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-data
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-r2
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-model
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-search
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-eval
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-cli-v2

clippy: prepare
	cd "$(WORKSPACE)" && $(CARGO) clippy --locked --all-targets $(PACKAGES) -- -D warnings

fmt: prepare
	cd "$(WORKSPACE)" && $(CARGO) fmt --all -- --check

release: test clippy fmt
