.PHONY: prepare parity

TRANSACTION := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOURCE_ARCHIVE := $(TRANSACTION)/source.tar
SOURCE_MANIFEST := $(TRANSACTION)/source-manifest.json
ARCHIVE_VERIFIER := $(TRANSACTION)/archive-verify.py
WORKSPACE := $(TMPDIR)/r2-map-rust-w4-workspace
TOOLCHAIN := /Users/john2/cascadia-bench/r2-map-v1/cache/runs/run-rust-verify-4dd80652629b716d-v6/rustup/toolchains/stable-aarch64-apple-darwin/bin
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

parity: prepare
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-r2 \
		mlx_export::tests::live_r2_map_capacity_is_the_rules_complete_23_tile_bound -- --exact --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-r2 \
		mlx_export::tests::wildlife_suffix_ -- --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-r2 \
		mlx_export::tests::authoritative_live_encoder_matches_independent_board_encoder_for_all_d6 -- --exact --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-r2 \
		model::tests::public_state_rejects_wildlife_on_a_zero_turn_starter_tile -- --exact --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-model \
		r2_map::tests::protocol_v3_emits_canonical_action_bytes_and_stable_request_identity -- --exact --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-model \
		r2_map::tests::rust_market_contract_matches_the_shared_compile_independent_fixture -- --exact --nocapture
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-search \
		r2_map_direct::tests::incremental_and_rayon_cache_match_sequential_oracle_byte_for_byte -- --exact --nocapture
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-search \
		r2_map_direct::tests::first_legal_93_token_replay_matches_authoritative_encoder_for_all_d6 -- --exact --nocapture
	cd "$(WORKSPACE)" && /usr/bin/time -p $(CARGO) test --locked -p cascadia-search \
		r2_map_direct::tests::staged_action_encoder_matches_canonical_for_market_variants_and_all_d6 -- --exact --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-search \
		r2_map_direct::tests::p1_sample_identity_binds_seed_game_index_transform_width_and_bin -- --exact --nocapture
	cd "$(WORKSPACE)" && $(CARGO) test --locked -p cascadia-search \
		r2_map_direct::tests::p1_corpus_digest_binds_per_game_and_total_action_counts -- --exact --nocapture
