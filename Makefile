.PHONY: bootstrap setup format format-check lint test cli-docs cli-docs-check performance-check performance-report mlx-device legacy-nnue-mlx-port legacy-nnue-mlx-convert legacy-nnue-mlx-fixture legacy-nnue-mlx-parity legacy-nnue-mlx-benchmark legacy-nnue-mlx-service legacy-nnue-mlx-exact-service legacy-nnue-mlx-rollout-parity legacy-nnue-mlx-gameplay-smoke legacy-nnue-mlx-gameplay-confirm rollout-value-smoke collect-rollout-value train-rollout-value resume-rollout-value rollout-joint-smoke collect-rollout-joint-validation train-rollout-joint resume-rollout-joint benchmark benchmark-research reproduce-research-requalification benchmark-smoke pattern-competition-smoke evaluate-pattern-competition pattern-portfolio-smoke evaluate-pattern-portfolio late-terminal-smoke late-terminal-confirm lookahead-benchmark lookahead-compare data-smoke ranking-data-smoke ranking-iteration-smoke terminal-ranking-smoke action-ranking-smoke imitation-smoke imitation-evidence-smoke imitation-evidence-parity conservative-advantage-smoke conservative-policy-smoke score-to-go-smoke score-to-go-hexgraph-smoke counterfactual-value-smoke counterfactual-advantage-smoke counterfactual-contrast-smoke collect-counterfactual-value-audit audit-counterfactual-value collect-counterfactual-advantage-audit audit-counterfactual-advantage collect-counterfactual-contrast-audit audit-counterfactual-contrast collect-r12-counterfactual-audit audit-r12-counterfactual collect-score-to-go-hexgraph-validation train-score-to-go-hexgraph benchmark-score-to-go-hexgraph resume-score-to-go-hexgraph public-beam-value-probe collect-public-beam-value collect-search collect-score-to-go collect-ranking collect-ranking-iteration collect-terminal-ranking enrich-action-ranking collect-action-ranking-test collect-imitation collect-imitation-evidence collect-imitation-score-residual-validation collect-imitation-test collect-conservative-advantage train-public-beam-value train-public-beam-set train-model train-search-value train-score-to-go resume-model train-smoke train-ranking train-ranking-iteration train-terminal-ranking train-action-ranking train-imitation train-imitation-distribution train-imitation-score-residual train-imitation-cross train-imitation-residual train-imitation-rotations train-conservative-advantage train-conservative-policy resume-ranking resume-imitation resume-imitation-distribution resume-imitation-score-residual ranking-train-smoke promote promote-public-beam-value promote-public-beam-set promote-search-value promote-ranking promote-ranking-iteration promote-terminal-ranking promote-action-ranking promote-imitation model-smoke public-beam-value-model-smoke public-beam-set-model-smoke evaluate-public-beam-value-test evaluate-public-beam-value evaluate-public-beam-set-test evaluate-model compare-model evaluate-ranking evaluate-ranking-iteration evaluate-terminal-ranking evaluate-action-ranking-test evaluate-imitation-test evaluate-conservative-advantage-test evaluate-conservative-policy-test evaluate-action-ranking evaluate-imitation evaluate-value-leaf web-dev web-build web-test build check
.PHONY: imitation-parent-residual-smoke collect-imitation-parent-train-priors collect-imitation-parent-validation collect-imitation-parent-validation-priors train-imitation-parent-residual resume-imitation-parent-residual
.PHONY: imitation-parent-hidden-smoke collect-imitation-parent-train-hidden collect-imitation-parent-hidden-validation collect-imitation-parent-validation-hidden train-imitation-parent-hidden resume-imitation-parent-hidden audit-imitation-identifiability
.PHONY: exact-mlx-crn-qualification exact-mlx-crn-smoke exact-mlx-crn-pilot exact-mlx-crn-confirm
.PHONY: counterfactual-ranker-smoke collect-r12-counterfactual-corpus train-r12-counterfactual-ranker resume-r12-counterfactual-ranker evaluate-r12-counterfactual-ranker evaluate-r12-counterfactual-test

V2_PACKAGES = -p cascadia-provenance -p cascadia-game -p cascadia-sim -p cascadia-eval -p cascadia-data -p cascadia-model -p cascadia-search -p cascadia-api -p cascadia-cli-v2
HOST_TOOL_PATH := /opt/homebrew/bin:/usr/local/bin:/opt/homebrew/opt/rustup/bin:/usr/local/opt/rustup/bin:$(PATH)
UV := $(or $(firstword $(wildcard /opt/homebrew/bin/uv /usr/local/bin/uv)),$(shell command -v uv 2>/dev/null),uv)
NPM_BIN := $(or $(firstword $(wildcard /opt/homebrew/bin/npm /usr/local/bin/npm)),$(shell command -v npm 2>/dev/null),npm)
NPM := /usr/bin/env PATH=$(HOST_TOOL_PATH) $(NPM_BIN)
CARGO := $(or $(firstword $(wildcard /opt/homebrew/opt/rustup/bin/cargo /usr/local/opt/rustup/bin/cargo)),$(shell command -v cargo 2>/dev/null),cargo)
RUN_DIR ?= artifacts/runs/entity-value-v1
MODEL_DIR ?= artifacts/models/entity-value-v1
TRAIN_DATASET ?= artifacts/datasets/greedy-v1-train-256
VALIDATION_DATASET ?= artifacts/datasets/greedy-v1-validation-64
EPOCHS ?= 10
MODEL_GAMES ?= 20
LOOKAHEAD_GAMES ?= 50
LOOKAHEAD_FIRST_SEED ?= 20400
RESEARCH_GAMES ?= 10
RESEARCH_FIRST_SEED ?= 30000
RANKING_RUN_DIR ?= artifacts/runs/entity-ranker-v1-h6
RANKING_MODEL_DIR ?= artifacts/models/entity-ranker-v1-h6
RANKING_TRAIN_DATASET ?= artifacts/datasets/ranking-h6-train
RANKING_VALIDATION_DATASET ?= artifacts/datasets/ranking-h6-validation
RANKING_TEACHER ?= habitat
RANKING_CANDIDATES ?= 8
RANKING_BEAR_CANDIDATES ?= 8
RANKING_HABITAT_CANDIDATES ?= 6
RANKING_EPOCHS ?= 20
RANKING_PATIENCE ?= 5
RANKING_GAMES ?= 20
RANKING_ITERATION_TRAIN_DATASET ?= artifacts/datasets/ranking-h6-iteration1-train
RANKING_ITERATION_VALIDATION_DATASET ?= artifacts/datasets/ranking-h6-iteration1-validation
RANKING_ITERATION_RUN_DIR ?= artifacts/runs/entity-ranker-v1-h6-iteration1
RANKING_ITERATION_MODEL_DIR ?= artifacts/models/entity-ranker-v1-h6-iteration1
RANKING_ITERATION_TRAIN_GAMES ?= 64
RANKING_ITERATION_VALIDATION_GAMES ?= 16
RANKING_ITERATION_EPOCHS ?= 10
SEARCH_VALUE_TRAIN_DATASET ?= artifacts/datasets/h6-value-train
SEARCH_VALUE_VALIDATION_DATASET ?= artifacts/datasets/h6-value-validation
SEARCH_VALUE_RUN_DIR ?= artifacts/runs/entity-value-v1-h6
SEARCH_VALUE_MODEL_DIR ?= artifacts/models/entity-value-v1-h6
TERMINAL_RANKING_TRAIN_DATASET ?= artifacts/datasets/ranking-terminal-r8-observable-train
TERMINAL_RANKING_VALIDATION_DATASET ?= artifacts/datasets/ranking-terminal-r8-observable-validation
TERMINAL_RANKING_RUN_DIR ?= artifacts/runs/entity-ranker-v2-terminal-r8-observable
TERMINAL_RANKING_MODEL_DIR ?= artifacts/models/entity-ranker-v2-terminal-r8-observable
TERMINAL_RANKING_TRAIN_GAMES ?= 64
TERMINAL_RANKING_VALIDATION_GAMES ?= 16
TERMINAL_RANKING_TRAIN_FIRST_GAME_INDEX ?= 64
TERMINAL_RANKING_VALIDATION_FIRST_GAME_INDEX ?= 16
TERMINAL_RANKING_EPOCHS ?= 20
ACTION_RANKING_TRAIN_DATASET ?= artifacts/datasets/action-ranking-terminal-r8-train
ACTION_RANKING_VALIDATION_DATASET ?= artifacts/datasets/action-ranking-terminal-r8-validation
ACTION_RANKING_TEST_SOURCE_DATASET ?= artifacts/datasets/ranking-terminal-r8-observable-test
ACTION_RANKING_TEST_DATASET ?= artifacts/datasets/action-ranking-terminal-r8-test
ACTION_RANKING_RUN_DIR ?= artifacts/runs/action-delta-ranker-v1-terminal-r8
ACTION_RANKING_MODEL_DIR ?= artifacts/models/action-delta-ranker-v1-terminal-r8
ACTION_RANKING_TEST_GAMES ?= 16
ACTION_RANKING_EPOCHS ?= 20
ACTION_RANKING_GAMES ?= 10
IMITATION_TRAIN_DATASET ?= artifacts/datasets/canonical-action-imitation-v1-train
IMITATION_VALIDATION_DATASET ?= artifacts/datasets/canonical-action-imitation-v1-validation
IMITATION_TEST_DATASET ?= artifacts/datasets/canonical-action-imitation-v1-test
IMITATION_RUN_DIR ?= artifacts/runs/canonical-action-imitation-v1
IMITATION_MODEL_DIR ?= artifacts/models/canonical-action-imitation-v1
IMITATION_CROSS_RUN_DIR ?= artifacts/runs/canonical-action-imitation-cross-v2-validation
IMITATION_RESIDUAL_RUN_DIR ?= artifacts/runs/canonical-action-imitation-residual-v2-validation
IMITATION_ROTATION_RUN_DIR ?= artifacts/runs/canonical-action-imitation-rotations-v2-validation
IMITATION_EVIDENCE_TRAIN_SOURCE ?= artifacts/datasets/canonical-action-mce-evidence-v1-train-actions
IMITATION_EVIDENCE_TRAIN_DATASET ?= artifacts/datasets/canonical-action-mce-evidence-v1-train-targets
IMITATION_EVIDENCE_VALIDATION_SOURCE ?= artifacts/datasets/canonical-action-mce-evidence-v1-validation-actions
IMITATION_EVIDENCE_VALIDATION_DATASET ?= artifacts/datasets/canonical-action-mce-evidence-v1-validation-targets
IMITATION_DISTRIBUTION_RUN_DIR ?= artifacts/runs/canonical-action-mce-distribution-v1-validation
IMITATION_SCORE_RESIDUAL_VALIDATION_SOURCE ?= artifacts/datasets/canonical-action-score-residual-v3-validation-actions
IMITATION_SCORE_RESIDUAL_VALIDATION_DATASET ?= artifacts/datasets/canonical-action-score-residual-v3-validation-targets
IMITATION_SCORE_RESIDUAL_RUN_DIR ?= artifacts/runs/canonical-action-score-residual-v3-validation
IMITATION_PARENT_TRAIN_PRIORS ?= artifacts/datasets/canonical-action-exact-parent-v1-train
IMITATION_PARENT_VALIDATION_SOURCE ?= artifacts/datasets/canonical-action-parent-residual-v4-validation-actions
IMITATION_PARENT_VALIDATION_TARGETS ?= artifacts/datasets/canonical-action-parent-residual-v4-validation-targets
IMITATION_PARENT_VALIDATION_PRIORS ?= artifacts/datasets/canonical-action-exact-parent-v1-validation
IMITATION_PARENT_RUN_DIR ?= artifacts/runs/exact-parent-candidate-set-residual-v4
IMITATION_PARENT_HIDDEN_TRAIN ?= artifacts/datasets/canonical-action-exact-parent-hidden-v1-train
IMITATION_PARENT_HIDDEN_VALIDATION_SOURCE ?= artifacts/datasets/canonical-action-parent-hidden-v5-validation-actions
IMITATION_PARENT_HIDDEN_VALIDATION_TARGETS ?= artifacts/datasets/canonical-action-parent-hidden-v5-validation-targets
IMITATION_PARENT_HIDDEN_VALIDATION ?= artifacts/datasets/canonical-action-exact-parent-hidden-v1-validation
IMITATION_PARENT_HIDDEN_RUN_DIR ?= artifacts/runs/exact-parent-hidden-state-residual-v5
IMITATION_EVIDENCE_FIRST_GAME_INDEX ?= 51000
IMITATION_EVIDENCE_TRAIN_GAMES ?= 64
IMITATION_EVIDENCE_VALIDATION_GAMES ?= 16
IMITATION_WEIGHTS ?= nnue_weights_v4opp_modal_iter3.bin
IMITATION_FIRST_GAME_INDEX ?= 50000
IMITATION_TRAIN_GAMES ?= 64
IMITATION_VALIDATION_GAMES ?= 16
IMITATION_TEST_GAMES ?= 16
IMITATION_EPOCHS ?= 20
IMITATION_GAMES ?= 10
CONSERVATIVE_ADVANTAGE_TRAIN_DATASET ?= artifacts/datasets/conservative-advantage-v1-train
CONSERVATIVE_ADVANTAGE_VALIDATION_DATASET ?= artifacts/datasets/conservative-advantage-v1-validation
CONSERVATIVE_ADVANTAGE_TEST_DATASET ?= artifacts/datasets/conservative-advantage-v1-test
CONSERVATIVE_ADVANTAGE_RUN_DIR ?= artifacts/runs/conservative-advantage-v1
CONSERVATIVE_ADVANTAGE_EPOCHS ?= 20
CONSERVATIVE_POLICY_RUN_DIR ?= artifacts/runs/conservative-policy-v2
CONSERVATIVE_POLICY_EPOCHS ?= 20
SCORE_TO_GO_TRAIN_DATASET ?= artifacts/datasets/score-to-go-h6-train
SCORE_TO_GO_VALIDATION_DATASET ?= artifacts/datasets/score-to-go-h6-validation
SCORE_TO_GO_RUN_DIR ?= artifacts/runs/score-to-go-h6-v1
SCORE_TO_GO_EPOCHS ?= 20
HEXGRAPH_SCORE_TO_GO_VALIDATION_DATASET ?= artifacts/datasets/score-to-go-hexgraph-v2-validation
HEXGRAPH_SCORE_TO_GO_TEST_DATASET ?= artifacts/datasets/score-to-go-hexgraph-v2-test
HEXGRAPH_SCORE_TO_GO_RUN_DIR ?= artifacts/runs/edge-aware-hex-score-to-go-v2
COUNTERFACTUAL_VALUE_AUDIT_DATASET ?= artifacts/datasets/counterfactual-public-value-audit-v1-validation
COUNTERFACTUAL_ADVANTAGE_AUDIT_DATASET ?= artifacts/datasets/same-decision-counterfactual-advantage-audit-v1-validation
COUNTERFACTUAL_CONTRAST_AUDIT_DATASET ?= artifacts/datasets/rank-stratified-counterfactual-contrast-audit-v1-validation
COUNTERFACTUAL_R12_AUDIT_DATASET ?= artifacts/datasets/r12-rank-stratified-estimator-audit-v1-validation
COUNTERFACTUAL_R12_TRAIN_DATASET ?= artifacts/datasets/r12-counterfactual-advantage-v1-train-128
COUNTERFACTUAL_R12_VALIDATION_DATASET ?= artifacts/datasets/r12-counterfactual-advantage-v1-validation-32
COUNTERFACTUAL_R12_TEST_DATASET ?= artifacts/datasets/r12-counterfactual-advantage-v1-test-32
COUNTERFACTUAL_R12_RUN_DIR ?= artifacts/runs/r12-counterfactual-advantage-set-ranker-v1
COUNTERFACTUAL_R12_TEST_AUTHORIZATION ?= $(COUNTERFACTUAL_R12_RUN_DIR)/test-authorization.json
PUBLIC_BEAM_VALUE_PROBE_DATASET ?= artifacts/datasets/public-beam-state-value-observability-v1
PUBLIC_BEAM_VALUE_TRAIN_DATASET ?= artifacts/datasets/public-beam-value-v1-train-32
PUBLIC_BEAM_VALUE_VALIDATION_DATASET ?= artifacts/datasets/public-beam-value-v1-validation-8
PUBLIC_BEAM_VALUE_TEST_DATASET ?= artifacts/datasets/public-beam-value-v1-test-8
PUBLIC_BEAM_VALUE_RUN_DIR ?= artifacts/runs/public-beam-value-v1
PUBLIC_BEAM_VALUE_MODEL_DIR ?= artifacts/models/public-beam-value-v1
PUBLIC_BEAM_SET_RUN_DIR ?= artifacts/runs/public-beam-set-ranker-v1
PUBLIC_BEAM_SET_MODEL_DIR ?= artifacts/models/public-beam-set-ranker-v1
LEGACY_NNUE_MLX_MODEL_DIR ?= artifacts/models/legacy-nnue-v4opp-mlx-v1
LEGACY_NNUE_MLX_FIXTURE ?= artifacts/fixtures/legacy-nnue-v4opp-mlx-v1-rust.json
LEGACY_NNUE_MLX_PARITY_REPORT ?= docs/v2/reports/legacy-nnue-v4opp-mlx-v1-parity.json
LEGACY_NNUE_MLX_BENCHMARK_REPORT ?= docs/v2/reports/legacy-nnue-v4opp-mlx-v1-benchmark.json
LEGACY_NNUE_MLX_SERVICE_DIRECT_REPORT ?= docs/v2/reports/legacy-nnue-v4opp-mlx-service-v1-direct.json
LEGACY_NNUE_MLX_SERVICE_REPORT ?= docs/v2/reports/legacy-nnue-v4opp-mlx-service-v1.json
LEGACY_NNUE_MLX_EXACT_SERVICE_REPORT ?= docs/v2/reports/legacy-nnue-v4opp-mlx-exact-csr-service-v1.json
LEGACY_NNUE_MLX_ROLLOUT_REPORT ?= docs/v2/reports/legacy-nnue-v4opp-mlx-exact-rollout-wave-v1.json
ROLLOUT_VALUE_SMOKE_DATASET ?= artifacts/datasets/exact-mlx-rollout-value-smoke-r32
ROLLOUT_VALUE_TRAIN_DATASET ?= artifacts/datasets/exact-mlx-rollout-value-r600-train
ROLLOUT_VALUE_VALIDATION_DATASET ?= artifacts/datasets/exact-mlx-rollout-value-r600-validation
ROLLOUT_VALUE_RUN_DIR ?= artifacts/runs/exact-mlx-rollout-return-v1
ROLLOUT_VALUE_MODEL_DIR ?= artifacts/models/exact-mlx-rollout-return-v1
ROLLOUT_JOINT_VALIDATION_DATASET ?= artifacts/datasets/exact-mlx-joint-ranking-r600-validation
ROLLOUT_JOINT_RUN_DIR ?= artifacts/runs/exact-mlx-joint-return-ranking-v1
ROLLOUT_JOINT_MODEL_DIR ?= artifacts/models/exact-mlx-joint-return-ranking-v1

bootstrap:
	./tools/bootstrap_macos.sh

setup:
	$(UV) sync --all-groups
	$(NPM) --prefix apps/web ci

format:
	$(CARGO) fmt --all
	$(UV) run ruff format python tests tools

format-check:
	$(CARGO) fmt --all -- --check
	$(UV) run ruff format --check python tests tools

lint:
	$(CARGO) clippy $(V2_PACKAGES) --all-targets --no-deps -- -D warnings
	$(UV) run ruff check python tests tools
	$(NPM) --prefix apps/web run lint

test:
	$(CARGO) test $(V2_PACKAGES) -p cascadia-differential
	$(UV) run pytest
	$(NPM) --prefix apps/web test

cli-docs:
	$(CARGO) build -p cascadia-cli-v2
	$(UV) run python tools/generate_cli_reference.py

cli-docs-check:
	$(CARGO) build -p cascadia-cli-v2
	$(UV) run python tools/generate_cli_reference.py --check

performance-check:
	$(UV) run python tools/verify_performance_budgets.py

performance-report:
	$(UV) run python tools/verify_performance_budgets.py \
		--output-json docs/v2/reports/v2-performance-qualification.json \
		--output-markdown docs/v2/reports/v2-performance-qualification.md

mlx-device:
	$(UV) run cascadia-mlx-device

legacy-nnue-mlx-port: legacy-nnue-mlx-convert legacy-nnue-mlx-fixture legacy-nnue-mlx-parity legacy-nnue-mlx-benchmark

legacy-nnue-mlx-convert:
	$(UV) run cascadia-mlx-legacy-nnue convert --source nnue_weights_v4opp_modal_iter3.bin --output $(LEGACY_NNUE_MLX_MODEL_DIR)

legacy-nnue-mlx-fixture:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher nnue-parity-fixture --games 1 --first-game-index 92000 --split train --weights nnue_weights_v4opp_modal_iter3.bin --output $(LEGACY_NNUE_MLX_FIXTURE)

legacy-nnue-mlx-parity:
	$(UV) run cascadia-mlx-legacy-nnue parity --source nnue_weights_v4opp_modal_iter3.bin --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --fixture $(LEGACY_NNUE_MLX_FIXTURE) --output $(LEGACY_NNUE_MLX_PARITY_REPORT) --synthetic-seed 20260619

legacy-nnue-mlx-benchmark:
	$(UV) run cascadia-mlx-legacy-nnue benchmark --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --output $(LEGACY_NNUE_MLX_BENCHMARK_REPORT) --seed 20260619 --iterations 200

legacy-nnue-mlx-service:
	$(UV) run cascadia-mlx-legacy-nnue service-parity --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --fixture $(LEGACY_NNUE_MLX_FIXTURE) --output $(LEGACY_NNUE_MLX_SERVICE_DIRECT_REPORT)
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	target/release/legacy-teacher nnue-service-parity --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --fixture $(LEGACY_NNUE_MLX_FIXTURE) --output $(LEGACY_NNUE_MLX_SERVICE_REPORT) --iterations 200

legacy-nnue-mlx-exact-service:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	target/release/legacy-teacher nnue-exact-service-parity --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --fixture $(LEGACY_NNUE_MLX_FIXTURE) --output $(LEGACY_NNUE_MLX_EXACT_SERVICE_REPORT) --iterations 200

legacy-nnue-mlx-rollout-parity:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher nnue-exact-rollout-wave-parity --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --fixture $(LEGACY_NNUE_MLX_FIXTURE) --weights $(IMITATION_WEIGHTS) --game-index 92100 --rollouts 32 --spot-decisions 0,39,79 --spot-rollouts 600 --output $(LEGACY_NNUE_MLX_ROLLOUT_REPORT)

exact-mlx-crn-qualification:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher exact-mlx-crn-compare --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --games 1 --first-seed 93012 --rollouts 32 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/exact-mlx-sequential-halving-crn-v1-r32-qualification.json

exact-mlx-crn-smoke:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher exact-mlx-crn-compare --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --games 1 --first-seed 35699 --rollouts 600 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/exact-mlx-sequential-halving-crn-v1-runtime-smoke-1.json

exact-mlx-crn-pilot:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher exact-mlx-crn-compare --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --games 3 --first-seed 35700 --rollouts 600 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/exact-mlx-sequential-halving-crn-v1-pilot3.json

exact-mlx-crn-confirm:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher exact-mlx-crn-confirm --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --games 20 --first-seed 35703 --rollouts 600 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/exact-mlx-sequential-halving-crn-v1-confirm20.json

legacy-nnue-mlx-gameplay-smoke:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher exact-mlx-productive-token-compare --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --games 1 --first-seed 32599 --rollouts 600 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/legacy-nnue-v4opp-exact-mlx-gameplay-smoke-1.json

legacy-nnue-mlx-gameplay-confirm:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher exact-mlx-productive-token-compare --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --games 10 --first-seed 32600 --rollouts 600 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/legacy-nnue-v4opp-exact-mlx-gameplay-confirm10.json

rollout-value-smoke:
	rm -rf $(ROLLOUT_VALUE_SMOKE_DATASET)
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-exact-mlx-rollout-values --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --output $(ROLLOUT_VALUE_SMOKE_DATASET) --games 1 --first-game-index 93000 --split train --rollouts 32 --trace-modulus 8 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-exact-mlx-rollout-values --dataset $(ROLLOUT_VALUE_SMOKE_DATASET)

collect-rollout-value:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-exact-mlx-rollout-values --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --output $(ROLLOUT_VALUE_TRAIN_DATASET) --games 4 --first-game-index 94000 --split train --resume --rollouts 600 --trace-modulus 8 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-exact-mlx-rollout-values --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --output $(ROLLOUT_VALUE_VALIDATION_DATASET) --games 2 --first-game-index 94000 --split validation --resume --rollouts 600 --trace-modulus 8 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-exact-mlx-rollout-values --dataset $(ROLLOUT_VALUE_TRAIN_DATASET)
	target/release/legacy-teacher validate-exact-mlx-rollout-values --dataset $(ROLLOUT_VALUE_VALIDATION_DATASET)

train-rollout-value:
	$(UV) run cascadia-mlx-rollout-value-train --parent-model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --train-dataset $(ROLLOUT_VALUE_TRAIN_DATASET) --validation-dataset $(ROLLOUT_VALUE_VALIDATION_DATASET) --run-dir $(ROLLOUT_VALUE_RUN_DIR) --derived-model-dir $(ROLLOUT_VALUE_MODEL_DIR)

resume-rollout-value:
	$(UV) run cascadia-mlx-rollout-value-train --parent-model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --train-dataset $(ROLLOUT_VALUE_TRAIN_DATASET) --validation-dataset $(ROLLOUT_VALUE_VALIDATION_DATASET) --run-dir $(ROLLOUT_VALUE_RUN_DIR) --derived-model-dir $(ROLLOUT_VALUE_MODEL_DIR) --resume

rollout-joint-smoke:
	rm -rf /tmp/cascadia-adr66-smoke-run /tmp/cascadia-adr66-smoke-model
	$(UV) run cascadia-mlx-rollout-joint-train --parent-model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --train-dataset $(ROLLOUT_VALUE_SMOKE_DATASET) --validation-dataset $(ROLLOUT_VALUE_SMOKE_DATASET) --run-dir /tmp/cascadia-adr66-smoke-run --derived-model-dir /tmp/cascadia-adr66-smoke-model --epochs 1 --implementation-smoke

collect-rollout-joint-validation:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-exact-mlx-rollout-values --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --output $(ROLLOUT_JOINT_VALIDATION_DATASET) --games 2 --first-game-index 95000 --split validation --resume --rollouts 600 --trace-modulus 8 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-exact-mlx-rollout-values --dataset $(ROLLOUT_JOINT_VALIDATION_DATASET)

train-rollout-joint:
	$(UV) run cascadia-mlx-rollout-joint-train --parent-model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --train-dataset $(ROLLOUT_VALUE_TRAIN_DATASET) --validation-dataset $(ROLLOUT_JOINT_VALIDATION_DATASET) --run-dir $(ROLLOUT_JOINT_RUN_DIR) --derived-model-dir $(ROLLOUT_JOINT_MODEL_DIR)

resume-rollout-joint:
	$(UV) run cascadia-mlx-rollout-joint-train --parent-model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --train-dataset $(ROLLOUT_VALUE_TRAIN_DATASET) --validation-dataset $(ROLLOUT_JOINT_VALIDATION_DATASET) --run-dir $(ROLLOUT_JOINT_RUN_DIR) --derived-model-dir $(ROLLOUT_JOINT_MODEL_DIR) --resume

benchmark:
	$(CARGO) run --release -p cascadia-cli-v2 -- benchmark --games 50 --strategy pattern-aware

benchmark-research:
	$(CARGO) run --release -p cascadia-cli-v2 -- late-conservative-base-policy-improvement-compare --games $(RESEARCH_GAMES) --first-seed $(RESEARCH_FIRST_SEED) --terminal-turns 5 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4 --sequential

reproduce-research-requalification:
	$(CARGO) run --release -p cascadia-cli-v2 -- late-conservative-base-policy-improvement-compare --games 50 --first-seed 35100 --terminal-turns 5 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4 --sequential

benchmark-smoke:
	$(CARGO) run --release -p cascadia-cli-v2 -- benchmark --games 4 --strategy random

pattern-competition-smoke:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 compare --games 1 --first-seed 25999 --baseline pattern-aware --treatment pattern-competition --output docs/v2/reports/pattern-competition-v1-runtime-smoke-1.json

evaluate-pattern-competition:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 compare --games 10 --first-seed 26000 --baseline pattern-aware --treatment pattern-competition --output docs/v2/reports/pattern-competition-v1-pilot10.json

pattern-portfolio-smoke:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 compare --games 1 --first-seed 26299 --baseline pattern-aware --treatment pattern-portfolio --sequential --output docs/v2/reports/pattern-portfolio-v1-runtime-smoke-1.json

evaluate-pattern-portfolio:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 compare --games 10 --first-seed 26300 --baseline pattern-aware --treatment pattern-portfolio --sequential --output docs/v2/reports/pattern-portfolio-v1-pilot10.json

late-terminal-smoke:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 late-terminal-policy-improvement-compare --games 1 --first-seed 26899 --terminal-turns 5 --determinizations 8 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4 --sequential --output docs/v2/reports/late-terminal-policy-improvement-v1-t5-r8-runtime-smoke-1.json

late-terminal-confirm:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 late-terminal-policy-improvement-compare --games 50 --first-seed 27000 --terminal-turns 5 --determinizations 8 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4 --sequential --output docs/v2/reports/late-terminal-policy-improvement-v1-t5-r8-confirm50.json

lookahead-benchmark:
	$(CARGO) run --release -p cascadia-cli-v2 -- lookahead-benchmark --games $(LOOKAHEAD_GAMES) --first-seed $(LOOKAHEAD_FIRST_SEED) --candidates 8 --determinizations 4 --greedy-plies 4

lookahead-compare:
	$(CARGO) run --release -p cascadia-cli-v2 -- lookahead-compare --games $(LOOKAHEAD_GAMES) --first-seed $(LOOKAHEAD_FIRST_SEED) --baseline greedy --candidates 8 --determinizations 4 --greedy-plies 4

data-smoke:
	rm -rf /tmp/cascadia-v2-data-smoke-train /tmp/cascadia-v2-data-smoke-validation
	$(CARGO) run -p cascadia-cli-v2 -- collect --output /tmp/cascadia-v2-data-smoke-train --games 1 --split train --strategy random --shard-games 1
	$(CARGO) run -p cascadia-cli-v2 -- collect --output /tmp/cascadia-v2-data-smoke-validation --games 1 --split validation --strategy random --shard-games 1
	$(CARGO) run -p cascadia-cli-v2 -- validate-dataset --dataset /tmp/cascadia-v2-data-smoke-train

ranking-data-smoke:
	rm -rf /tmp/cascadia-v2-ranking-smoke-train /tmp/cascadia-v2-ranking-smoke-validation
	$(CARGO) run --release -p cascadia-cli-v2 -- collect-ranking --teacher habitat --output /tmp/cascadia-v2-ranking-smoke-train --games 1 --split train --shard-games 1 --candidates 4 --habitat-candidates 2 --determinizations 1 --greedy-plies 1
	$(CARGO) run --release -p cascadia-cli-v2 -- collect-ranking --teacher habitat --output /tmp/cascadia-v2-ranking-smoke-validation --games 1 --split validation --shard-games 1 --candidates 4 --habitat-candidates 2 --determinizations 1 --greedy-plies 1
	$(CARGO) run --release -p cascadia-cli-v2 -- validate-ranking-dataset --dataset /tmp/cascadia-v2-ranking-smoke-train

ranking-iteration-smoke:
	rm -rf /tmp/cascadia-v2-iteration-smoke-base-train /tmp/cascadia-v2-iteration-smoke-train /tmp/cascadia-v2-iteration-smoke-validation /tmp/cascadia-v2-iteration-smoke-regression /tmp/cascadia-v2-iteration-smoke-run /tmp/cascadia-v2-iteration-smoke-model
	target/release/cascadia-v2 collect-ranking --teacher habitat --output /tmp/cascadia-v2-iteration-smoke-base-train --games 1 --first-game-index 9001 --split train --shard-games 1 --candidates 2 --habitat-candidates 1 --determinizations 1 --greedy-plies 1
	target/release/cascadia-v2 collect-ranking-iteration --output /tmp/cascadia-v2-iteration-smoke-train --games 1 --first-game-index 9000 --split train --shard-games 1 --model-dir $(RANKING_MODEL_DIR) --candidates 2 --habitat-candidates 1 --determinizations 1 --greedy-plies 1
	target/release/cascadia-v2 collect-ranking-iteration --output /tmp/cascadia-v2-iteration-smoke-validation --games 1 --first-game-index 9000 --split validation --shard-games 1 --model-dir $(RANKING_MODEL_DIR) --candidates 2 --habitat-candidates 1 --determinizations 1 --greedy-plies 1
	target/release/cascadia-v2 collect-ranking --teacher habitat --output /tmp/cascadia-v2-iteration-smoke-regression --games 1 --first-game-index 9001 --split validation --shard-games 1 --candidates 2 --habitat-candidates 1 --determinizations 1 --greedy-plies 1
	$(UV) run cascadia-mlx-ranking-train --train-dataset /tmp/cascadia-v2-iteration-smoke-base-train --additional-train-dataset /tmp/cascadia-v2-iteration-smoke-train --validation-dataset /tmp/cascadia-v2-iteration-smoke-validation --regression-validation-dataset /tmp/cascadia-v2-iteration-smoke-regression --run-dir /tmp/cascadia-v2-iteration-smoke-run --init-model-dir $(RANKING_MODEL_DIR) --epochs 1 --group-batch-size 16 --checkpoint-steps 2 --validation-patience 2 --learning-rate 0.00003
	$(UV) run cascadia-mlx-ranking-promote --run-dir /tmp/cascadia-v2-iteration-smoke-run --output /tmp/cascadia-v2-iteration-smoke-model
	target/release/cascadia-v2 habitat-ranking-model-h2h --games 1 --first-seed 23999 --baseline-model-dir $(RANKING_MODEL_DIR) --treatment-model-dir /tmp/cascadia-v2-iteration-smoke-model --candidates 2 --habitat-candidates 1

terminal-ranking-smoke:
	rm -rf /tmp/cascadia-v2-terminal-ranking-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-terminal-ranking --output /tmp/cascadia-v2-terminal-ranking-smoke/train --games 1 --first-game-index 9900 --split train --shard-games 1 --determinizations 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 collect-terminal-ranking --output /tmp/cascadia-v2-terminal-ranking-smoke/validation --games 1 --first-game-index 9900 --split validation --shard-games 1 --determinizations 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 validate-ranking-dataset --dataset /tmp/cascadia-v2-terminal-ranking-smoke/train
	target/release/cascadia-v2 validate-ranking-dataset --dataset /tmp/cascadia-v2-terminal-ranking-smoke/validation
	$(UV) run cascadia-mlx-ranking-train --train-dataset /tmp/cascadia-v2-terminal-ranking-smoke/train --validation-dataset /tmp/cascadia-v2-terminal-ranking-smoke/validation --run-dir /tmp/cascadia-v2-terminal-ranking-smoke/run --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --validation-patience 2
	$(UV) run cascadia-mlx-ranking-promote --run-dir /tmp/cascadia-v2-terminal-ranking-smoke/run --output /tmp/cascadia-v2-terminal-ranking-smoke/model
	target/release/cascadia-v2 pattern-ranking-model-compare --games 1 --first-seed 25099 --model-dir /tmp/cascadia-v2-terminal-ranking-smoke/model --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2

action-ranking-smoke:
	rm -rf /tmp/cascadia-v2-action-ranking-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-terminal-ranking --output /tmp/cascadia-v2-action-ranking-smoke/source-train --games 1 --first-game-index 9910 --split train --shard-games 1 --determinizations 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 collect-terminal-ranking --output /tmp/cascadia-v2-action-ranking-smoke/source-validation --games 1 --first-game-index 9910 --split validation --shard-games 1 --determinizations 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 collect-terminal-ranking --output /tmp/cascadia-v2-action-ranking-smoke/source-test --games 1 --first-game-index 9910 --split test --shard-games 1 --determinizations 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 enrich-action-ranking --source-dataset /tmp/cascadia-v2-action-ranking-smoke/source-train --output /tmp/cascadia-v2-action-ranking-smoke/train --policy-market-draws 2
	target/release/cascadia-v2 enrich-action-ranking --source-dataset /tmp/cascadia-v2-action-ranking-smoke/source-validation --output /tmp/cascadia-v2-action-ranking-smoke/validation --policy-market-draws 2
	target/release/cascadia-v2 enrich-action-ranking --source-dataset /tmp/cascadia-v2-action-ranking-smoke/source-test --output /tmp/cascadia-v2-action-ranking-smoke/test --policy-market-draws 2
	target/release/cascadia-v2 validate-action-ranking-dataset --dataset /tmp/cascadia-v2-action-ranking-smoke/train
	$(UV) run cascadia-mlx-action-ranking-train --train-dataset /tmp/cascadia-v2-action-ranking-smoke/train --validation-dataset /tmp/cascadia-v2-action-ranking-smoke/validation --run-dir /tmp/cascadia-v2-action-ranking-smoke/run --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --validation-patience 2
	$(UV) run cascadia-mlx-action-ranking-evaluate --run-dir /tmp/cascadia-v2-action-ranking-smoke/run --test-dataset /tmp/cascadia-v2-action-ranking-smoke/test --group-batch-size 8
	target/release/cascadia-v2 action-ranking-model-compare --run-dir /tmp/cascadia-v2-action-ranking-smoke/run --games 1 --first-seed 25699 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2

imitation-smoke:
	rm -rf /tmp/cascadia-v2-imitation-smoke
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	$(CARGO) build --release -p cascadia-cli-v2
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation --output /tmp/cascadia-v2-imitation-smoke/train --games 1 --first-game-index 90000 --split train --shard-games 1 --group-limit 64 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation --output /tmp/cascadia-v2-imitation-smoke/validation --games 1 --first-game-index 90000 --split validation --shard-games 1 --group-limit 64 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-dataset --dataset /tmp/cascadia-v2-imitation-smoke/train
	$(UV) run cascadia-mlx-imitation-train --train-dataset /tmp/cascadia-v2-imitation-smoke/train --validation-dataset /tmp/cascadia-v2-imitation-smoke/validation --run-dir /tmp/cascadia-v2-imitation-smoke/run --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --validation-patience 2
	target/release/cascadia-v2 full-action-imitation-compare --run-dir /tmp/cascadia-v2-imitation-smoke/run --games 1 --first-seed 90000 --output /tmp/cascadia-v2-imitation-smoke/gameplay.json

imitation-evidence-smoke:
	rm -rf /tmp/cascadia-v2-imitation-evidence-smoke
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output /tmp/cascadia-v2-imitation-evidence-smoke/train-actions --targets-output /tmp/cascadia-v2-imitation-evidence-smoke/train-targets --games 1 --first-game-index 90000 --split train --group-limit 96 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output /tmp/cascadia-v2-imitation-evidence-smoke/validation-actions --targets-output /tmp/cascadia-v2-imitation-evidence-smoke/validation-targets --games 1 --first-game-index 90000 --split validation --group-limit 96 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	$(UV) run cascadia-mlx-imitation-distribution-train --train-dataset /tmp/cascadia-v2-imitation-evidence-smoke/train-targets --validation-dataset /tmp/cascadia-v2-imitation-evidence-smoke/validation-targets --run-dir /tmp/cascadia-v2-imitation-evidence-smoke/run --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --validation-patience 2

imitation-parent-residual-smoke:
	rm -rf /tmp/cascadia-v2-parent-residual-smoke
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output /tmp/cascadia-v2-parent-residual-smoke/train-actions --targets-output /tmp/cascadia-v2-parent-residual-smoke/train-targets --games 1 --first-game-index 90010 --split train --group-limit 96 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output /tmp/cascadia-v2-parent-residual-smoke/validation-actions --targets-output /tmp/cascadia-v2-parent-residual-smoke/validation-targets --games 1 --first-game-index 90010 --split validation --group-limit 96 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-priors --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset /tmp/cascadia-v2-parent-residual-smoke/train-targets --output /tmp/cascadia-v2-parent-residual-smoke/train-priors
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-priors --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset /tmp/cascadia-v2-parent-residual-smoke/validation-targets --output /tmp/cascadia-v2-parent-residual-smoke/validation-priors
	$(UV) run cascadia-mlx-imitation-parent-residual-train --train-dataset /tmp/cascadia-v2-parent-residual-smoke/train-priors --validation-dataset /tmp/cascadia-v2-parent-residual-smoke/validation-priors --run-dir /tmp/cascadia-v2-parent-residual-smoke/run --epochs 1 --validation-patience 2

imitation-parent-hidden-smoke:
	rm -rf /tmp/cascadia-v2-parent-hidden-smoke
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output /tmp/cascadia-v2-parent-hidden-smoke/train-actions --targets-output /tmp/cascadia-v2-parent-hidden-smoke/train-targets --games 1 --first-game-index 90011 --split train --group-limit 96 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output /tmp/cascadia-v2-parent-hidden-smoke/validation-actions --targets-output /tmp/cascadia-v2-parent-hidden-smoke/validation-targets --games 1 --first-game-index 90011 --split validation --group-limit 96 --immediate-limit 16 --rollouts 2 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-hidden --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset /tmp/cascadia-v2-parent-hidden-smoke/train-targets --output /tmp/cascadia-v2-parent-hidden-smoke/train-hidden
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-hidden --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset /tmp/cascadia-v2-parent-hidden-smoke/validation-targets --output /tmp/cascadia-v2-parent-hidden-smoke/validation-hidden
	$(UV) run cascadia-mlx-imitation-parent-hidden-train --train-dataset /tmp/cascadia-v2-parent-hidden-smoke/train-hidden --validation-dataset /tmp/cascadia-v2-parent-hidden-smoke/validation-hidden --run-dir /tmp/cascadia-v2-parent-hidden-smoke/run --epochs 1 --validation-patience 2

imitation-evidence-parity:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher teacher-estimate-parity --games 1 --first-game-index 90000 --split train --rollouts 600 --weights $(IMITATION_WEIGHTS) --output docs/v2/reports/canonical-action-teacher-estimate-parity-r600.json

conservative-advantage-smoke:
	rm -rf /tmp/cascadia-v2-conservative-advantage-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-conservative-advantage --output /tmp/cascadia-v2-conservative-advantage-smoke/train --games 1 --first-game-index 9990 --split train --shard-games 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 collect-conservative-advantage --output /tmp/cascadia-v2-conservative-advantage-smoke/validation --games 1 --first-game-index 9990 --split validation --shard-games 1 --policy-candidates 2 --policy-habitat-candidates 1 --policy-bear-candidates 1 --policy-market-draws 2
	target/release/cascadia-v2 validate-conservative-advantage-dataset --dataset /tmp/cascadia-v2-conservative-advantage-smoke/train
	$(UV) run cascadia-mlx-conservative-advantage-train --train-dataset /tmp/cascadia-v2-conservative-advantage-smoke/train --validation-dataset /tmp/cascadia-v2-conservative-advantage-smoke/validation --run-dir /tmp/cascadia-v2-conservative-advantage-smoke/run --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --validation-patience 2

conservative-policy-smoke: conservative-advantage-smoke
	rm -rf /tmp/cascadia-v2-conservative-policy-smoke
	$(UV) run cascadia-mlx-conservative-policy-train --train-dataset /tmp/cascadia-v2-conservative-advantage-smoke/train --validation-dataset /tmp/cascadia-v2-conservative-advantage-smoke/validation --run-dir /tmp/cascadia-v2-conservative-policy-smoke --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --validation-patience 2

score-to-go-smoke:
	rm -rf /tmp/cascadia-v2-score-to-go-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-score-to-go --output /tmp/cascadia-v2-score-to-go-smoke/train --games 1 --first-game-index 9991 --split train --shard-games 1
	target/release/cascadia-v2 collect-score-to-go --output /tmp/cascadia-v2-score-to-go-smoke/validation --games 1 --first-game-index 9991 --split validation --shard-games 1
	target/release/cascadia-v2 validate-score-to-go-dataset --dataset /tmp/cascadia-v2-score-to-go-smoke/train
	$(UV) run cascadia-mlx-score-to-go-train --train-dataset /tmp/cascadia-v2-score-to-go-smoke/train --validation-dataset /tmp/cascadia-v2-score-to-go-smoke/validation --run-dir /tmp/cascadia-v2-score-to-go-smoke/run --epochs 1 --batch-size 32 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0

score-to-go-hexgraph-smoke:
	rm -rf /tmp/cascadia-v2-score-to-go-hexgraph-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-score-to-go --output /tmp/cascadia-v2-score-to-go-hexgraph-smoke/train --games 1 --first-game-index 9992 --split train --shard-games 1
	target/release/cascadia-v2 collect-score-to-go --output /tmp/cascadia-v2-score-to-go-hexgraph-smoke/validation --games 1 --first-game-index 9992 --split validation --shard-games 1
	target/release/cascadia-v2 validate-score-to-go-dataset --dataset /tmp/cascadia-v2-score-to-go-hexgraph-smoke/train
	target/release/cascadia-v2 validate-score-to-go-dataset --dataset /tmp/cascadia-v2-score-to-go-hexgraph-smoke/validation
	$(UV) run cascadia-mlx-score-to-go-train --train-dataset /tmp/cascadia-v2-score-to-go-hexgraph-smoke/train --validation-dataset /tmp/cascadia-v2-score-to-go-hexgraph-smoke/validation --run-dir /tmp/cascadia-v2-score-to-go-hexgraph-smoke/run --epochs 1 --batch-size 80 --checkpoint-steps 1 --validation-patience 2 --architecture edge-aware-hex-score-to-go-v2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --graph-blocks 2 --market-blocks 0 --hex-rotation-augmentation

counterfactual-value-smoke:
	rm -rf /tmp/cascadia-v2-counterfactual-value-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-value --output /tmp/cascadia-v2-counterfactual-value-smoke --games 1 --first-game-index 9993 --split train --samples-per-state 2
	target/release/cascadia-v2 validate-counterfactual-value-dataset --dataset /tmp/cascadia-v2-counterfactual-value-smoke
	target/release/cascadia-v2 audit-counterfactual-value-dataset --dataset /tmp/cascadia-v2-counterfactual-value-smoke --output docs/v2/reports/counterfactual-public-value-target-audit-v1-implementation-smoke.json

counterfactual-advantage-smoke:
	rm -rf /tmp/cascadia-v2-counterfactual-advantage-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output /tmp/cascadia-v2-counterfactual-advantage-smoke --games 1 --first-game-index 9994 --split train --groups-per-game 4 --samples-per-candidate 2
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset /tmp/cascadia-v2-counterfactual-advantage-smoke
	target/release/cascadia-v2 audit-counterfactual-advantage-dataset --dataset /tmp/cascadia-v2-counterfactual-advantage-smoke --output docs/v2/reports/same-decision-counterfactual-advantage-target-audit-v1-implementation-smoke.json --markdown-output docs/v2/reports/same-decision-counterfactual-advantage-target-audit-v1-implementation-smoke.md

counterfactual-contrast-smoke:
	rm -rf /tmp/cascadia-v2-counterfactual-contrast-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output /tmp/cascadia-v2-counterfactual-contrast-smoke --games 1 --first-game-index 9995 --split train --groups-per-game 4 --samples-per-candidate 2 --candidate-selection stratified
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset /tmp/cascadia-v2-counterfactual-contrast-smoke
	target/release/cascadia-v2 audit-counterfactual-advantage-dataset --dataset /tmp/cascadia-v2-counterfactual-contrast-smoke --output docs/v2/reports/rank-stratified-counterfactual-contrast-audit-v1-implementation-smoke.json --markdown-output docs/v2/reports/rank-stratified-counterfactual-contrast-audit-v1-implementation-smoke.md

counterfactual-ranker-smoke:
	rm -rf /tmp/cascadia-v2-counterfactual-ranker-smoke
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output /tmp/cascadia-v2-counterfactual-ranker-smoke/train --games 1 --first-game-index 9996 --split train --groups-per-game 4 --samples-per-candidate 12 --candidate-selection stratified
	target/release/cascadia-v2 collect-counterfactual-advantage --output /tmp/cascadia-v2-counterfactual-ranker-smoke/validation --games 1 --first-game-index 9997 --split validation --groups-per-game 4 --samples-per-candidate 12 --candidate-selection stratified
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/train
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/validation
	$(UV) run cascadia-mlx-counterfactual-advantage-train --train-dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/train --validation-dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/validation --run-dir /tmp/cascadia-v2-counterfactual-ranker-smoke/run --epochs 1 --group-batch-size 4 --checkpoint-steps 1 --validation-patience 2
	$(UV) run cascadia-mlx-counterfactual-advantage-train --train-dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/train --validation-dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/validation --run-dir /tmp/cascadia-v2-counterfactual-ranker-smoke/run --epochs 2 --group-batch-size 4 --checkpoint-steps 1 --validation-patience 2 --resume
	$(UV) run cascadia-mlx-counterfactual-advantage-evaluate --run-dir /tmp/cascadia-v2-counterfactual-ranker-smoke/run --dataset /tmp/cascadia-v2-counterfactual-ranker-smoke/validation --output docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-implementation-smoke.json --markdown-output docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-implementation-smoke.md --group-batch-size 4

collect-ranking:
	$(CARGO) run --release -p cascadia-cli-v2 -- collect-ranking --teacher $(RANKING_TEACHER) --output $(RANKING_TRAIN_DATASET) --games 128 --split train --shard-games 8 --resume --candidates $(RANKING_CANDIDATES) --bear-candidates $(RANKING_BEAR_CANDIDATES) --habitat-candidates $(RANKING_HABITAT_CANDIDATES) --determinizations 4 --greedy-plies 4
	$(CARGO) run --release -p cascadia-cli-v2 -- collect-ranking --teacher $(RANKING_TEACHER) --output $(RANKING_VALIDATION_DATASET) --games 32 --split validation --shard-games 8 --resume --candidates $(RANKING_CANDIDATES) --bear-candidates $(RANKING_BEAR_CANDIDATES) --habitat-candidates $(RANKING_HABITAT_CANDIDATES) --determinizations 4 --greedy-plies 4
	$(CARGO) run --release -p cascadia-cli-v2 -- validate-ranking-dataset --dataset $(RANKING_TRAIN_DATASET)
	$(CARGO) run --release -p cascadia-cli-v2 -- validate-ranking-dataset --dataset $(RANKING_VALIDATION_DATASET)

collect-imitation:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation --output $(IMITATION_TRAIN_DATASET) --games $(IMITATION_TRAIN_GAMES) --first-game-index $(IMITATION_FIRST_GAME_INDEX) --split train --shard-games 1 --resume --group-limit 64 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation --output $(IMITATION_VALIDATION_DATASET) --games $(IMITATION_VALIDATION_GAMES) --first-game-index $(IMITATION_FIRST_GAME_INDEX) --split validation --shard-games 1 --resume --group-limit 64 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-dataset --dataset $(IMITATION_TRAIN_DATASET)
	target/release/legacy-teacher validate-imitation-dataset --dataset $(IMITATION_VALIDATION_DATASET)

collect-imitation-evidence:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output $(IMITATION_EVIDENCE_TRAIN_SOURCE) --targets-output $(IMITATION_EVIDENCE_TRAIN_DATASET) --games $(IMITATION_EVIDENCE_TRAIN_GAMES) --first-game-index $(IMITATION_EVIDENCE_FIRST_GAME_INDEX) --split train --resume --group-limit 96 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output $(IMITATION_EVIDENCE_VALIDATION_SOURCE) --targets-output $(IMITATION_EVIDENCE_VALIDATION_DATASET) --games $(IMITATION_EVIDENCE_VALIDATION_GAMES) --first-game-index $(IMITATION_EVIDENCE_FIRST_GAME_INDEX) --split validation --resume --group-limit 96 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-targets --dataset $(IMITATION_EVIDENCE_TRAIN_DATASET)
	target/release/legacy-teacher validate-imitation-targets --dataset $(IMITATION_EVIDENCE_VALIDATION_DATASET)

collect-imitation-score-residual-validation:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output $(IMITATION_SCORE_RESIDUAL_VALIDATION_SOURCE) --targets-output $(IMITATION_SCORE_RESIDUAL_VALIDATION_DATASET) --games 16 --first-game-index 51016 --split validation --resume --group-limit 96 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-targets --dataset $(IMITATION_SCORE_RESIDUAL_VALIDATION_DATASET)

collect-imitation-parent-train-priors:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-priors --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset $(IMITATION_EVIDENCE_TRAIN_DATASET) --output $(IMITATION_PARENT_TRAIN_PRIORS) --resume
	target/release/legacy-teacher validate-imitation-parent-priors --dataset $(IMITATION_PARENT_TRAIN_PRIORS)

collect-imitation-parent-validation:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output $(IMITATION_PARENT_VALIDATION_SOURCE) --targets-output $(IMITATION_PARENT_VALIDATION_TARGETS) --games 16 --first-game-index 51032 --split validation --resume --group-limit 96 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-targets --dataset $(IMITATION_PARENT_VALIDATION_TARGETS)

collect-imitation-parent-validation-priors:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-priors --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset $(IMITATION_PARENT_VALIDATION_TARGETS) --output $(IMITATION_PARENT_VALIDATION_PRIORS) --resume
	target/release/legacy-teacher validate-imitation-parent-priors --dataset $(IMITATION_PARENT_VALIDATION_PRIORS)

collect-imitation-parent-train-hidden:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-hidden --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset $(IMITATION_EVIDENCE_TRAIN_DATASET) --output $(IMITATION_PARENT_HIDDEN_TRAIN) --resume
	target/release/legacy-teacher validate-imitation-parent-hidden --dataset $(IMITATION_PARENT_HIDDEN_TRAIN)

collect-imitation-parent-hidden-validation:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-evidence --source-output $(IMITATION_PARENT_HIDDEN_VALIDATION_SOURCE) --targets-output $(IMITATION_PARENT_HIDDEN_VALIDATION_TARGETS) --games 16 --first-game-index 51048 --split validation --resume --group-limit 96 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-targets --dataset $(IMITATION_PARENT_HIDDEN_VALIDATION_TARGETS)

collect-imitation-parent-validation-hidden:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation-parent-hidden --model-dir $(LEGACY_NNUE_MLX_MODEL_DIR) --source-dataset $(IMITATION_PARENT_HIDDEN_VALIDATION_TARGETS) --output $(IMITATION_PARENT_HIDDEN_VALIDATION) --resume
	target/release/legacy-teacher validate-imitation-parent-hidden --dataset $(IMITATION_PARENT_HIDDEN_VALIDATION)

collect-imitation-test:
	$(CARGO) build --release -p cascadia-differential --features legacy-teacher --bin legacy-teacher
	MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 target/release/legacy-teacher collect-imitation --output $(IMITATION_TEST_DATASET) --games $(IMITATION_TEST_GAMES) --first-game-index $(IMITATION_FIRST_GAME_INDEX) --split test --shard-games 1 --resume --group-limit 64 --immediate-limit 16 --rollouts 600 --weights $(IMITATION_WEIGHTS)
	target/release/legacy-teacher validate-imitation-dataset --dataset $(IMITATION_TEST_DATASET)

collect-ranking-iteration:
	target/release/cascadia-v2 collect-ranking-iteration --output $(RANKING_ITERATION_TRAIN_DATASET) --games $(RANKING_ITERATION_TRAIN_GAMES) --first-game-index 128 --split train --shard-games 8 --resume --model-dir $(RANKING_MODEL_DIR) --candidates 8 --habitat-candidates 6 --determinizations 4 --greedy-plies 4
	target/release/cascadia-v2 collect-ranking-iteration --output $(RANKING_ITERATION_VALIDATION_DATASET) --games $(RANKING_ITERATION_VALIDATION_GAMES) --first-game-index 32 --split validation --shard-games 8 --resume --model-dir $(RANKING_MODEL_DIR) --candidates 8 --habitat-candidates 6 --determinizations 4 --greedy-plies 4
	target/release/cascadia-v2 validate-ranking-dataset --dataset $(RANKING_ITERATION_TRAIN_DATASET)
	target/release/cascadia-v2 validate-ranking-dataset --dataset $(RANKING_ITERATION_VALIDATION_DATASET)

collect-terminal-ranking:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-terminal-ranking --output $(TERMINAL_RANKING_TRAIN_DATASET) --games $(TERMINAL_RANKING_TRAIN_GAMES) --first-game-index $(TERMINAL_RANKING_TRAIN_FIRST_GAME_INDEX) --split train --shard-games 1 --resume --determinizations 8 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4
	target/release/cascadia-v2 collect-terminal-ranking --output $(TERMINAL_RANKING_VALIDATION_DATASET) --games $(TERMINAL_RANKING_VALIDATION_GAMES) --first-game-index $(TERMINAL_RANKING_VALIDATION_FIRST_GAME_INDEX) --split validation --shard-games 1 --resume --determinizations 8 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4
	target/release/cascadia-v2 validate-ranking-dataset --dataset $(TERMINAL_RANKING_TRAIN_DATASET)
	target/release/cascadia-v2 validate-ranking-dataset --dataset $(TERMINAL_RANKING_VALIDATION_DATASET)

enrich-action-ranking:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 enrich-action-ranking --source-dataset $(TERMINAL_RANKING_TRAIN_DATASET) --output $(ACTION_RANKING_TRAIN_DATASET) --resume --policy-market-draws 4
	target/release/cascadia-v2 enrich-action-ranking --source-dataset $(TERMINAL_RANKING_VALIDATION_DATASET) --output $(ACTION_RANKING_VALIDATION_DATASET) --resume --policy-market-draws 4
	target/release/cascadia-v2 validate-action-ranking-dataset --dataset $(ACTION_RANKING_TRAIN_DATASET)
	target/release/cascadia-v2 validate-action-ranking-dataset --dataset $(ACTION_RANKING_VALIDATION_DATASET)

collect-action-ranking-test:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-terminal-ranking --output $(ACTION_RANKING_TEST_SOURCE_DATASET) --games $(ACTION_RANKING_TEST_GAMES) --first-game-index 0 --split test --shard-games 1 --resume --determinizations 8 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4
	target/release/cascadia-v2 validate-ranking-dataset --dataset $(ACTION_RANKING_TEST_SOURCE_DATASET)
	target/release/cascadia-v2 enrich-action-ranking --source-dataset $(ACTION_RANKING_TEST_SOURCE_DATASET) --output $(ACTION_RANKING_TEST_DATASET) --resume --policy-market-draws 4
	target/release/cascadia-v2 validate-action-ranking-dataset --dataset $(ACTION_RANKING_TEST_DATASET)

collect-conservative-advantage:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-conservative-advantage --output $(CONSERVATIVE_ADVANTAGE_TRAIN_DATASET) --games 128 --first-game-index 0 --split train --shard-games 1 --resume --terminal-turns 5 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4
	target/release/cascadia-v2 collect-conservative-advantage --output $(CONSERVATIVE_ADVANTAGE_VALIDATION_DATASET) --games 32 --first-game-index 0 --split validation --shard-games 1 --resume --terminal-turns 5 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4
	target/release/cascadia-v2 collect-conservative-advantage --output $(CONSERVATIVE_ADVANTAGE_TEST_DATASET) --games 32 --first-game-index 0 --split test --shard-games 1 --resume --terminal-turns 5 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4
	target/release/cascadia-v2 validate-conservative-advantage-dataset --dataset $(CONSERVATIVE_ADVANTAGE_TRAIN_DATASET)
	target/release/cascadia-v2 validate-conservative-advantage-dataset --dataset $(CONSERVATIVE_ADVANTAGE_VALIDATION_DATASET)
	target/release/cascadia-v2 validate-conservative-advantage-dataset --dataset $(CONSERVATIVE_ADVANTAGE_TEST_DATASET)

public-beam-value-probe:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 public-beam-value-probe --output $(PUBLIC_BEAM_VALUE_PROBE_DATASET) --first-game-index 40000 --games 2 --resume --report docs/v2/reports/public-beam-state-value-observability-v1-r8x2-b16-w2.json
	target/release/cascadia-v2 validate-public-beam-value-dataset --dataset $(PUBLIC_BEAM_VALUE_PROBE_DATASET)

collect-public-beam-value:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-public-beam-value --output $(PUBLIC_BEAM_VALUE_TRAIN_DATASET) --games 32 --first-game-index 41000 --split train --resume
	target/release/cascadia-v2 collect-public-beam-value --output $(PUBLIC_BEAM_VALUE_VALIDATION_DATASET) --games 8 --first-game-index 41000 --split validation --resume
	target/release/cascadia-v2 collect-public-beam-value --output $(PUBLIC_BEAM_VALUE_TEST_DATASET) --games 8 --first-game-index 41000 --split test --resume
	target/release/cascadia-v2 validate-public-beam-value-dataset --dataset $(PUBLIC_BEAM_VALUE_TRAIN_DATASET)
	target/release/cascadia-v2 validate-public-beam-value-dataset --dataset $(PUBLIC_BEAM_VALUE_VALIDATION_DATASET)
	target/release/cascadia-v2 validate-public-beam-value-dataset --dataset $(PUBLIC_BEAM_VALUE_TEST_DATASET)

collect-search:
	target/release/cascadia-v2 collect-search --output $(SEARCH_VALUE_TRAIN_DATASET) --games 256 --split train --shard-games 8 --resume --candidates 8 --habitat-candidates 6 --determinizations 4 --greedy-plies 4
	target/release/cascadia-v2 collect-search --output $(SEARCH_VALUE_VALIDATION_DATASET) --games 64 --split validation --shard-games 8 --resume --candidates 8 --habitat-candidates 6 --determinizations 4 --greedy-plies 4
	target/release/cascadia-v2 validate-dataset --dataset $(SEARCH_VALUE_TRAIN_DATASET)
	target/release/cascadia-v2 validate-dataset --dataset $(SEARCH_VALUE_VALIDATION_DATASET)

collect-score-to-go:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-score-to-go --output $(SCORE_TO_GO_TRAIN_DATASET) --games 256 --first-game-index 0 --split train --shard-games 1 --resume
	target/release/cascadia-v2 collect-score-to-go --output $(SCORE_TO_GO_VALIDATION_DATASET) --games 64 --first-game-index 0 --split validation --shard-games 1 --resume
	target/release/cascadia-v2 validate-score-to-go-dataset --dataset $(SCORE_TO_GO_TRAIN_DATASET)
	target/release/cascadia-v2 validate-score-to-go-dataset --dataset $(SCORE_TO_GO_VALIDATION_DATASET)

collect-score-to-go-hexgraph-validation:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-score-to-go --output $(HEXGRAPH_SCORE_TO_GO_VALIDATION_DATASET) --games 64 --first-game-index 64000 --split validation --shard-games 1 --resume
	target/release/cascadia-v2 validate-score-to-go-dataset --dataset $(HEXGRAPH_SCORE_TO_GO_VALIDATION_DATASET)

collect-counterfactual-value-audit:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-value --output $(COUNTERFACTUAL_VALUE_AUDIT_DATASET) --games 2 --first-game-index 65000 --split validation --samples-per-state 16 --resume
	target/release/cascadia-v2 validate-counterfactual-value-dataset --dataset $(COUNTERFACTUAL_VALUE_AUDIT_DATASET)

audit-counterfactual-value:
	target/release/cascadia-v2 audit-counterfactual-value-dataset --dataset $(COUNTERFACTUAL_VALUE_AUDIT_DATASET) --output docs/v2/reports/counterfactual-public-value-target-audit-v1.json

collect-counterfactual-advantage-audit:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output $(COUNTERFACTUAL_ADVANTAGE_AUDIT_DATASET) --games 2 --first-game-index 66000 --split validation --groups-per-game 16 --samples-per-candidate 16 --resume
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_ADVANTAGE_AUDIT_DATASET)

audit-counterfactual-advantage:
	target/release/cascadia-v2 audit-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_ADVANTAGE_AUDIT_DATASET) --output docs/v2/reports/same-decision-counterfactual-advantage-target-audit-v1.json --markdown-output docs/v2/reports/same-decision-counterfactual-advantage-target-audit-v1.md

collect-counterfactual-contrast-audit:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output $(COUNTERFACTUAL_CONTRAST_AUDIT_DATASET) --games 2 --first-game-index 67000 --split validation --groups-per-game 16 --samples-per-candidate 16 --candidate-selection stratified --resume
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_CONTRAST_AUDIT_DATASET)

audit-counterfactual-contrast:
	target/release/cascadia-v2 audit-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_CONTRAST_AUDIT_DATASET) --output docs/v2/reports/rank-stratified-counterfactual-contrast-audit-v1.json --markdown-output docs/v2/reports/rank-stratified-counterfactual-contrast-audit-v1.md

collect-r12-counterfactual-audit:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output $(COUNTERFACTUAL_R12_AUDIT_DATASET) --games 2 --first-game-index 68000 --split validation --groups-per-game 16 --samples-per-candidate 16 --candidate-selection stratified --resume
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_R12_AUDIT_DATASET)

audit-r12-counterfactual:
	target/release/cascadia-v2 audit-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_R12_AUDIT_DATASET) --output docs/v2/reports/r12-rank-stratified-estimator-audit-v1.json --markdown-output docs/v2/reports/r12-rank-stratified-estimator-audit-v1.md --estimator-samples 12

collect-r12-counterfactual-corpus:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 collect-counterfactual-advantage --output $(COUNTERFACTUAL_R12_TRAIN_DATASET) --games 128 --first-game-index 69000 --split train --groups-per-game 16 --samples-per-candidate 12 --candidate-selection stratified --resume
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_R12_TRAIN_DATASET)
	target/release/cascadia-v2 collect-counterfactual-advantage --output $(COUNTERFACTUAL_R12_VALIDATION_DATASET) --games 32 --first-game-index 70000 --split validation --groups-per-game 16 --samples-per-candidate 12 --candidate-selection stratified --resume
	target/release/cascadia-v2 validate-counterfactual-advantage-dataset --dataset $(COUNTERFACTUAL_R12_VALIDATION_DATASET)

train-r12-counterfactual-ranker:
	$(UV) run cascadia-mlx-counterfactual-advantage-train --train-dataset $(COUNTERFACTUAL_R12_TRAIN_DATASET) --validation-dataset $(COUNTERFACTUAL_R12_VALIDATION_DATASET) --run-dir $(COUNTERFACTUAL_R12_RUN_DIR) --epochs 20 --group-batch-size 32 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260614 --checkpoint-steps 100 --validation-patience 5

resume-r12-counterfactual-ranker:
	$(UV) run cascadia-mlx-counterfactual-advantage-train --train-dataset $(COUNTERFACTUAL_R12_TRAIN_DATASET) --validation-dataset $(COUNTERFACTUAL_R12_VALIDATION_DATASET) --run-dir $(COUNTERFACTUAL_R12_RUN_DIR) --epochs 20 --group-batch-size 32 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260614 --checkpoint-steps 100 --validation-patience 5 --resume

evaluate-r12-counterfactual-ranker:
	$(UV) run cascadia-mlx-counterfactual-advantage-evaluate --run-dir $(COUNTERFACTUAL_R12_RUN_DIR) --dataset $(COUNTERFACTUAL_R12_VALIDATION_DATASET) --output docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-validation.json --markdown-output docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-validation.md --group-batch-size 32

evaluate-r12-counterfactual-test:
	PYTHONPATH=python uv run python tools/adr0079_counterfactual_advantage_test.py --run-dir $(COUNTERFACTUAL_R12_RUN_DIR) --dataset $(COUNTERFACTUAL_R12_TEST_DATASET) --validation-report $(COUNTERFACTUAL_R12_RUN_DIR)/validation-report.json --authorization $(COUNTERFACTUAL_R12_TEST_AUTHORIZATION) --output docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-test.json --markdown-output docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-test.md --group-batch-size 32

train-model:
	$(UV) run cascadia-mlx-train --train-dataset $(TRAIN_DATASET) --validation-dataset $(VALIDATION_DATASET) --run-dir $(RUN_DIR) --epochs $(EPOCHS) --batch-size 256

train-search-value:
	$(UV) run cascadia-mlx-train --train-dataset $(SEARCH_VALUE_TRAIN_DATASET) --validation-dataset $(SEARCH_VALUE_VALIDATION_DATASET) --run-dir $(SEARCH_VALUE_RUN_DIR) --epochs $(EPOCHS) --batch-size 256 --learning-rate 0.0003 --weight-decay 0.0001

train-score-to-go:
	$(UV) run cascadia-mlx-score-to-go-train --train-dataset $(SCORE_TO_GO_TRAIN_DATASET) --validation-dataset $(SCORE_TO_GO_VALIDATION_DATASET) --run-dir $(SCORE_TO_GO_RUN_DIR) --epochs $(SCORE_TO_GO_EPOCHS) --batch-size 256 --learning-rate 0.0003 --weight-decay 0.0001

train-score-to-go-hexgraph:
	$(UV) run cascadia-mlx-score-to-go-train --train-dataset $(SCORE_TO_GO_TRAIN_DATASET) --validation-dataset $(HEXGRAPH_SCORE_TO_GO_VALIDATION_DATASET) --run-dir $(HEXGRAPH_SCORE_TO_GO_RUN_DIR) --epochs 30 --batch-size 256 --learning-rate 0.0003 --weight-decay 0.0001 --seed 20260624 --checkpoint-steps 500 --validation-patience 6 --baseline-run-dir $(SCORE_TO_GO_RUN_DIR) --architecture edge-aware-hex-score-to-go-v2 --hidden-dim 96 --attention-heads 4 --board-blocks 0 --graph-blocks 4 --market-blocks 1 --hex-rotation-augmentation

benchmark-score-to-go-hexgraph:
	$(UV) run cascadia-mlx-score-to-go-benchmark --run-dir $(HEXGRAPH_SCORE_TO_GO_RUN_DIR) --dataset $(HEXGRAPH_SCORE_TO_GO_VALIDATION_DATASET) --output docs/v2/reports/edge-aware-hex-score-to-go-v2-inference.json --batch-size 256 --warmup-iterations 10 --iterations 100

resume-score-to-go-hexgraph:
	$(UV) run cascadia-mlx-score-to-go-train --train-dataset $(SCORE_TO_GO_TRAIN_DATASET) --validation-dataset $(HEXGRAPH_SCORE_TO_GO_VALIDATION_DATASET) --run-dir $(HEXGRAPH_SCORE_TO_GO_RUN_DIR) --epochs 30 --batch-size 256 --learning-rate 0.0003 --weight-decay 0.0001 --seed 20260624 --checkpoint-steps 500 --validation-patience 6 --baseline-run-dir $(SCORE_TO_GO_RUN_DIR) --architecture edge-aware-hex-score-to-go-v2 --hidden-dim 96 --attention-heads 4 --board-blocks 0 --graph-blocks 4 --market-blocks 1 --hex-rotation-augmentation --resume

resume-model:
	$(UV) run cascadia-mlx-train --train-dataset $(TRAIN_DATASET) --validation-dataset $(VALIDATION_DATASET) --run-dir $(RUN_DIR) --epochs $(EPOCHS) --batch-size 256 --resume

train-ranking:
	$(UV) run cascadia-mlx-ranking-train --train-dataset $(RANKING_TRAIN_DATASET) --validation-dataset $(RANKING_VALIDATION_DATASET) --run-dir $(RANKING_RUN_DIR) --epochs $(RANKING_EPOCHS) --group-batch-size 16 --validation-patience $(RANKING_PATIENCE)

train-ranking-iteration:
	$(UV) run cascadia-mlx-ranking-train --train-dataset $(RANKING_TRAIN_DATASET) --additional-train-dataset $(RANKING_ITERATION_TRAIN_DATASET) --validation-dataset $(RANKING_ITERATION_VALIDATION_DATASET) --regression-validation-dataset $(RANKING_VALIDATION_DATASET) --run-dir $(RANKING_ITERATION_RUN_DIR) --init-model-dir $(RANKING_MODEL_DIR) --epochs $(RANKING_ITERATION_EPOCHS) --group-batch-size 16 --validation-patience 3 --learning-rate 0.00003 --weight-decay 0.0001

train-terminal-ranking:
	$(UV) run cascadia-mlx-ranking-train --train-dataset $(TERMINAL_RANKING_TRAIN_DATASET) --validation-dataset $(TERMINAL_RANKING_VALIDATION_DATASET) --run-dir $(TERMINAL_RANKING_RUN_DIR) --epochs $(TERMINAL_RANKING_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001

train-action-ranking:
	$(UV) run cascadia-mlx-action-ranking-train --train-dataset $(ACTION_RANKING_TRAIN_DATASET) --validation-dataset $(ACTION_RANKING_VALIDATION_DATASET) --run-dir $(ACTION_RANKING_RUN_DIR) --epochs $(ACTION_RANKING_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001

train-imitation:
	$(UV) run cascadia-mlx-imitation-train --train-dataset $(IMITATION_TRAIN_DATASET) --validation-dataset $(IMITATION_VALIDATION_DATASET) --run-dir $(IMITATION_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001

train-imitation-distribution:
	$(UV) run cascadia-mlx-imitation-distribution-train --train-dataset $(IMITATION_EVIDENCE_TRAIN_DATASET) --validation-dataset $(IMITATION_EVIDENCE_VALIDATION_DATASET) --run-dir $(IMITATION_DISTRIBUTION_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260616

train-imitation-score-residual:
	$(UV) run cascadia-mlx-imitation-score-residual-train --train-dataset $(IMITATION_EVIDENCE_TRAIN_DATASET) --validation-dataset $(IMITATION_SCORE_RESIDUAL_VALIDATION_DATASET) --run-dir $(IMITATION_SCORE_RESIDUAL_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260617

train-imitation-parent-residual:
	$(UV) run cascadia-mlx-imitation-parent-residual-train --train-dataset $(IMITATION_PARENT_TRAIN_PRIORS) --validation-dataset $(IMITATION_PARENT_VALIDATION_PRIORS) --run-dir $(IMITATION_PARENT_RUN_DIR)

train-imitation-parent-hidden:
	$(UV) run cascadia-mlx-imitation-parent-hidden-train --train-dataset $(IMITATION_PARENT_HIDDEN_TRAIN) --validation-dataset $(IMITATION_PARENT_HIDDEN_VALIDATION) --run-dir $(IMITATION_PARENT_HIDDEN_RUN_DIR)

train-imitation-cross:
	$(UV) run cascadia-mlx-imitation-train --train-dataset $(IMITATION_TRAIN_DATASET) --validation-dataset $(IMITATION_VALIDATION_DATASET) --run-dir $(IMITATION_CROSS_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260613 --architecture shared-state-action-cross-ranker-v2

train-imitation-residual:
	$(UV) run cascadia-mlx-imitation-train --train-dataset $(IMITATION_TRAIN_DATASET) --validation-dataset $(IMITATION_VALIDATION_DATASET) --run-dir $(IMITATION_RESIDUAL_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260614 --architecture shared-state-action-residual-ranker-v2 --immediate-rank-prior 0.08

train-imitation-rotations:
	$(UV) run cascadia-mlx-imitation-train --train-dataset $(IMITATION_TRAIN_DATASET) --validation-dataset $(IMITATION_VALIDATION_DATASET) --run-dir $(IMITATION_ROTATION_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260615 --hex-rotation-augmentation

train-conservative-advantage:
	$(UV) run cascadia-mlx-conservative-advantage-train --train-dataset $(CONSERVATIVE_ADVANTAGE_TRAIN_DATASET) --validation-dataset $(CONSERVATIVE_ADVANTAGE_VALIDATION_DATASET) --run-dir $(CONSERVATIVE_ADVANTAGE_RUN_DIR) --epochs $(CONSERVATIVE_ADVANTAGE_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001

train-conservative-policy:
	$(UV) run cascadia-mlx-conservative-policy-train --train-dataset $(CONSERVATIVE_ADVANTAGE_TRAIN_DATASET) --validation-dataset $(CONSERVATIVE_ADVANTAGE_VALIDATION_DATASET) --run-dir $(CONSERVATIVE_POLICY_RUN_DIR) --epochs $(CONSERVATIVE_POLICY_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001

train-public-beam-value:
	$(UV) run cascadia-mlx-public-beam-value-train --train-dataset $(PUBLIC_BEAM_VALUE_TRAIN_DATASET) --validation-dataset $(PUBLIC_BEAM_VALUE_VALIDATION_DATASET) --run-dir $(PUBLIC_BEAM_VALUE_RUN_DIR) --epochs 20 --group-batch-size 8 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001

train-public-beam-set:
	$(UV) run cascadia-mlx-public-beam-set-train --train-dataset $(PUBLIC_BEAM_VALUE_TRAIN_DATASET) --validation-dataset $(PUBLIC_BEAM_VALUE_VALIDATION_DATASET) --run-dir $(PUBLIC_BEAM_SET_RUN_DIR) --epochs 20 --group-batch-size 8 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260613

resume-ranking:
	$(UV) run cascadia-mlx-ranking-train --train-dataset $(RANKING_TRAIN_DATASET) --validation-dataset $(RANKING_VALIDATION_DATASET) --run-dir $(RANKING_RUN_DIR) --epochs $(RANKING_EPOCHS) --group-batch-size 16 --validation-patience $(RANKING_PATIENCE) --resume

resume-imitation:
	$(UV) run cascadia-mlx-imitation-train --train-dataset $(IMITATION_TRAIN_DATASET) --validation-dataset $(IMITATION_VALIDATION_DATASET) --run-dir $(IMITATION_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --resume

resume-imitation-distribution:
	$(UV) run cascadia-mlx-imitation-distribution-train --train-dataset $(IMITATION_EVIDENCE_TRAIN_DATASET) --validation-dataset $(IMITATION_EVIDENCE_VALIDATION_DATASET) --run-dir $(IMITATION_DISTRIBUTION_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260616 --resume

resume-imitation-score-residual:
	$(UV) run cascadia-mlx-imitation-score-residual-train --train-dataset $(IMITATION_EVIDENCE_TRAIN_DATASET) --validation-dataset $(IMITATION_SCORE_RESIDUAL_VALIDATION_DATASET) --run-dir $(IMITATION_SCORE_RESIDUAL_RUN_DIR) --epochs $(IMITATION_EPOCHS) --group-batch-size 16 --validation-patience 5 --learning-rate 0.0001 --weight-decay 0.0001 --seed 20260617 --resume

resume-imitation-parent-residual:
	$(UV) run cascadia-mlx-imitation-parent-residual-train --train-dataset $(IMITATION_PARENT_TRAIN_PRIORS) --validation-dataset $(IMITATION_PARENT_VALIDATION_PRIORS) --run-dir $(IMITATION_PARENT_RUN_DIR) --resume

resume-imitation-parent-hidden:
	$(UV) run cascadia-mlx-imitation-parent-hidden-train --train-dataset $(IMITATION_PARENT_HIDDEN_TRAIN) --validation-dataset $(IMITATION_PARENT_HIDDEN_VALIDATION) --run-dir $(IMITATION_PARENT_HIDDEN_RUN_DIR) --resume

audit-imitation-identifiability:
	$(UV) run cascadia-mlx-imitation-identifiability --dataset $(IMITATION_PARENT_HIDDEN_TRAIN) --output docs/v2/reports/mce-teacher-identifiability-train.json
	$(UV) run cascadia-mlx-imitation-identifiability --dataset $(IMITATION_PARENT_HIDDEN_VALIDATION) --output docs/v2/reports/mce-teacher-identifiability-validation.json

train-smoke: data-smoke
	rm -rf /tmp/cascadia-v2-train-smoke
	rm -rf /tmp/cascadia-v2-model-smoke
	$(UV) run cascadia-mlx-train --train-dataset /tmp/cascadia-v2-data-smoke-train --validation-dataset /tmp/cascadia-v2-data-smoke-validation --run-dir /tmp/cascadia-v2-train-smoke --epochs 1 --batch-size 16 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 1 --market-blocks 1
	$(UV) run cascadia-mlx-train --train-dataset /tmp/cascadia-v2-data-smoke-train --validation-dataset /tmp/cascadia-v2-data-smoke-validation --run-dir /tmp/cascadia-v2-train-smoke --epochs 2 --batch-size 16 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 1 --market-blocks 1 --resume
	$(UV) run cascadia-mlx-promote --run-dir /tmp/cascadia-v2-train-smoke --output /tmp/cascadia-v2-model-smoke
	$(CARGO) run -p cascadia-cli-v2 -- model-smoke --model-dir /tmp/cascadia-v2-model-smoke

ranking-train-smoke: ranking-data-smoke
	rm -rf /tmp/cascadia-v2-ranking-run-smoke
	rm -rf /tmp/cascadia-v2-ranking-model-smoke
	$(UV) run cascadia-mlx-ranking-train --train-dataset /tmp/cascadia-v2-ranking-smoke-train --validation-dataset /tmp/cascadia-v2-ranking-smoke-validation --run-dir /tmp/cascadia-v2-ranking-run-smoke --epochs 1 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0
	$(UV) run cascadia-mlx-ranking-train --train-dataset /tmp/cascadia-v2-ranking-smoke-train --validation-dataset /tmp/cascadia-v2-ranking-smoke-validation --run-dir /tmp/cascadia-v2-ranking-run-smoke --epochs 2 --group-batch-size 8 --checkpoint-steps 2 --hidden-dim 32 --attention-heads 4 --board-blocks 0 --market-blocks 0 --resume
	$(UV) run cascadia-mlx-ranking-promote --run-dir /tmp/cascadia-v2-ranking-run-smoke --output /tmp/cascadia-v2-ranking-model-smoke
	$(CARGO) run --release -p cascadia-cli-v2 -- ranking-model-benchmark --model-dir /tmp/cascadia-v2-ranking-model-smoke --games 1

promote:
	$(UV) run cascadia-mlx-promote --run-dir $(RUN_DIR) --output $(MODEL_DIR)

promote-search-value:
	$(UV) run cascadia-mlx-promote --run-dir $(SEARCH_VALUE_RUN_DIR) --output $(SEARCH_VALUE_MODEL_DIR)

promote-ranking:
	$(UV) run cascadia-mlx-ranking-promote --run-dir $(RANKING_RUN_DIR) --output $(RANKING_MODEL_DIR)

promote-ranking-iteration:
	$(UV) run cascadia-mlx-ranking-promote --run-dir $(RANKING_ITERATION_RUN_DIR) --output $(RANKING_ITERATION_MODEL_DIR)

promote-terminal-ranking:
	$(UV) run cascadia-mlx-ranking-promote --run-dir $(TERMINAL_RANKING_RUN_DIR) --output $(TERMINAL_RANKING_MODEL_DIR)

promote-action-ranking:
	$(UV) run cascadia-mlx-action-ranking-promote --run-dir $(ACTION_RANKING_RUN_DIR) --output $(ACTION_RANKING_MODEL_DIR)

promote-imitation:
	$(UV) run cascadia-mlx-imitation-promote --run-dir $(IMITATION_RUN_DIR) --output $(IMITATION_MODEL_DIR)

promote-public-beam-value:
	$(UV) run cascadia-mlx-public-beam-value-promote --run-dir $(PUBLIC_BEAM_VALUE_RUN_DIR) --output $(PUBLIC_BEAM_VALUE_MODEL_DIR)

promote-public-beam-set:
	$(UV) run cascadia-mlx-public-beam-set-promote --run-dir $(PUBLIC_BEAM_SET_RUN_DIR) --output $(PUBLIC_BEAM_SET_MODEL_DIR)

model-smoke:
	$(CARGO) run -p cascadia-cli-v2 -- model-smoke --model-dir $(MODEL_DIR)

public-beam-value-model-smoke:
	target/release/cascadia-v2 public-beam-value-model-smoke --run-dir $(PUBLIC_BEAM_VALUE_RUN_DIR)

public-beam-set-model-smoke:
	target/release/cascadia-v2 public-beam-value-model-smoke --run-dir $(PUBLIC_BEAM_SET_RUN_DIR) --server .venv/bin/cascadia-mlx-public-beam-set-serve

evaluate-model:
	$(CARGO) run --release -p cascadia-cli-v2 -- model-benchmark --games $(MODEL_GAMES) --model-dir $(MODEL_DIR)

compare-model:
	$(CARGO) run --release -p cascadia-cli-v2 -- model-compare --games $(MODEL_GAMES) --first-seed 10000 --baseline greedy --model-dir $(MODEL_DIR)

evaluate-ranking:
	$(CARGO) run --release -p cascadia-cli-v2 -- ranking-habitat-prefilter-compare --model-dir $(RANKING_MODEL_DIR) --games $(RANKING_GAMES) --first-seed 22500 --baseline-candidates 8 --baseline-habitat-candidates 6 --candidates 16 --habitat-candidates 8 --immediate-anchors 8 --prefilter-candidates 14 --determinizations 4 --greedy-plies 4

evaluate-ranking-iteration:
	target/release/cascadia-v2 habitat-ranking-model-h2h --baseline-model-dir $(RANKING_MODEL_DIR) --treatment-model-dir $(RANKING_ITERATION_MODEL_DIR) --games $(RANKING_GAMES) --first-seed 23700 --candidates 8 --habitat-candidates 6

evaluate-terminal-ranking:
	target/release/cascadia-v2 pattern-ranking-model-compare --model-dir $(TERMINAL_RANKING_MODEL_DIR) --games $(RANKING_GAMES) --first-seed 25400 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4 --output docs/v2/reports/entity-ranker-v2-terminal-r8-observable-pilot.json

evaluate-action-ranking-test:
	$(UV) run cascadia-mlx-action-ranking-evaluate --run-dir $(ACTION_RANKING_RUN_DIR) --test-dataset $(ACTION_RANKING_TEST_DATASET) --group-batch-size 16

evaluate-imitation-test:
	$(UV) run cascadia-mlx-imitation-evaluate --run-dir $(IMITATION_RUN_DIR) --test-dataset $(IMITATION_TEST_DATASET) --group-batch-size 16

evaluate-conservative-advantage-test:
	$(UV) run cascadia-mlx-conservative-advantage-evaluate --run-dir $(CONSERVATIVE_ADVANTAGE_RUN_DIR) --test-dataset $(CONSERVATIVE_ADVANTAGE_TEST_DATASET) --group-batch-size 16

evaluate-conservative-policy-test:
	$(UV) run cascadia-mlx-conservative-policy-evaluate --run-dir $(CONSERVATIVE_POLICY_RUN_DIR) --test-dataset $(CONSERVATIVE_ADVANTAGE_TEST_DATASET) --group-batch-size 16

evaluate-public-beam-value-test:
	$(UV) run cascadia-mlx-public-beam-value-evaluate --run-dir $(PUBLIC_BEAM_VALUE_RUN_DIR) --test-dataset $(PUBLIC_BEAM_VALUE_TEST_DATASET) --group-batch-size 8

evaluate-public-beam-value:
	target/release/cascadia-v2 public-beam-value-model-compare --model-dir $(PUBLIC_BEAM_VALUE_MODEL_DIR) --games 10 --first-seed 31000 --output docs/v2/reports/mlx-public-beam-value-v1-vs-strong-pilot10.json

evaluate-public-beam-set-test:
	$(UV) run cascadia-mlx-public-beam-set-evaluate --run-dir $(PUBLIC_BEAM_SET_RUN_DIR) --test-dataset $(PUBLIC_BEAM_VALUE_TEST_DATASET) --group-batch-size 8

evaluate-public-beam-set:
	target/release/cascadia-v2 public-beam-value-model-compare --model-dir $(PUBLIC_BEAM_SET_MODEL_DIR) --server .venv/bin/cascadia-mlx-public-beam-set-serve --games 10 --first-seed 31000 --output docs/v2/reports/mlx-public-beam-set-ranker-v1-vs-strong-pilot10.json

evaluate-action-ranking:
	target/release/cascadia-v2 action-ranking-model-compare --model-dir $(ACTION_RANKING_MODEL_DIR) --games $(ACTION_RANKING_GAMES) --first-seed 25700 --policy-candidates 8 --policy-habitat-candidates 6 --policy-bear-candidates 8 --policy-market-draws 4 --output docs/v2/reports/action-delta-ranker-v1-terminal-r8-pilot.json

evaluate-imitation:
	$(CARGO) build --release -p cascadia-cli-v2
	target/release/cascadia-v2 full-action-imitation-compare --model-dir $(IMITATION_MODEL_DIR) --games $(IMITATION_GAMES) --first-seed 32700 --output docs/v2/reports/canonical-action-imitation-v1-pilot10.json

evaluate-value-leaf:
	target/release/cascadia-v2 value-leaf-compare --model-dir $(SEARCH_VALUE_MODEL_DIR) --games $(MODEL_GAMES) --first-seed 22700 --candidates 8 --habitat-candidates 6 --determinizations 4 --greedy-plies 4

web-dev:
	./tools/web_dev.sh

web-build:
	$(NPM) --prefix apps/web run build
	$(CARGO) build --release -p cascadia-api

web-test:
	$(NPM) --prefix apps/web run lint
	$(NPM) --prefix apps/web test
	$(NPM) --prefix apps/web run test:e2e

build:
	$(NPM) --prefix apps/web run build
	$(CARGO) build --release $(V2_PACKAGES)

check: format-check lint test cli-docs-check performance-check web-test
