.DEFAULT_GOAL := audit
.PHONY: prepare audit

TRANSACTION := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOURCE_ARCHIVE := $(TRANSACTION)/source.tar
SOURCE_MANIFEST := $(TRANSACTION)/source-manifest.json
ARCHIVE_VERIFIER := $(TRANSACTION)/archive-verify.py
WORKSPACE := $(TMPDIR)/r2-map-rust-p1-workspace
TOOLCHAIN := /Users/john2/cascadia-bench/r2-map-v1/cache/runs/run-rust-verify-4dd80652629b716d-v6/rustup/toolchains/stable-aarch64-apple-darwin/bin
CARGO := /usr/bin/env PATH="$(TOOLCHAIN):$$PATH" CARGO_TERM_COLOR=never cargo
P1_GAMES ?= 100
P1_CALIBRATION ?= 0

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

audit: prepare
	cd "$(WORKSPACE)" && /usr/bin/python3 tools/r2_map_p1_resource_gate.py \
		--metrics "$(WORKSPACE)/p1-time-metrics.txt" -- /usr/bin/env \
		R2_MAP_P1_GAMES="$(P1_GAMES)" \
		R2_MAP_P1_ALLOW_CALIBRATION="$(P1_CALIBRATION)" \
		$(CARGO) test --release --locked -p cascadia-search \
		r2_map_direct::tests::incremental_open_corpus_complete_game_gate -- --exact --ignored --nocapture
