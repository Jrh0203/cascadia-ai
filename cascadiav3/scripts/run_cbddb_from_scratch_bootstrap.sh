#!/usr/bin/env bash
set -euo pipefail

# CBDDB EI-0: TRUE from-scratch (random-init) search bootstrap.
#
# This is the first CascadiaFormer bootstrap under the CBDDB scoring-card ruleset
# (cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19), trained from a
# random init rather than warm-started from any AAAAA/greedy-retention weights.
#
# Roots come from CBDDB greedy self-play states, each retained K32 menu is
# labeled by sampled greedy rollouts, and the actual trajectory advances by the
# greedy action to avoid the rollout-state distribution-shift failure. Scoring
# cards are threaded end-to-end via SCORING_CARDS=cbddb so self-play config,
# tensor feature extraction, and shard ruleset tagging are all CBDDB-matched.
#
# Sizing is economical for a ~3-GPU-day budget (300 train / 50 val seed blocks,
# 15k steps). Every value is overridable via env so the orchestrator can tune
# seed counts / steps without editing this file.

export JOB_SLUG="${JOB_SLUG:-cbddb_from_scratch_bootstrap}"
export PROFILE="${PROFILE:-cbddb_from_scratch_bootstrap}"

# CBDDB ruleset threaded through every exporter generation call.
export SCORING_CARDS="${SCORING_CARDS:-cbddb}"

# Same generation/objective/filter recipe as the AAAAA EI-0 greedy-search
# bootstrap (docs/v3/TRAINING_PIPELINE.md "Stage 2: EI-0 Search Bootstrap").
export EXPERT_TENSOR_MODE="${EXPERT_TENSOR_MODE:-greedy_search_bootstrap}"
export MAX_ACTIONS="${MAX_ACTIONS:-32}"
export FILTER_TOP_K="${FILTER_TOP_K:-32}"
export FILTER_MODE="${FILTER_MODE:-greedy-prefix-strict}"
export OBJECTIVE="${OBJECTIVE:-search-improved-greedy-retention}"
export SELECTION_METRIC="${SELECTION_METRIC:-locked_val_total}"
export SELECTION_MODE="${SELECTION_MODE:-min}"
export MODEL_SIZE="${MODEL_SIZE:-S}"

# Fresh seed blocks, never previously used. Kept distinct from the 2027195000+
# range reserved for certification.
export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2027193000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-300}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2027193500}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-50}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-80}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"

# TRUE random init (no warm start). The pipeline maps INIT_MANIFEST="none" to an
# empty init so the trainer starts from a fresh random model. LR uses the
# doc's from-scratch fallback (2e-4) rather than the 1e-4 warm-start value.
export INIT_MANIFEST="${INIT_MANIFEST:-none}"
export LR="${LR:-0.0002}"

# Economical step budget for ~3 GPU-days; override for a longer run.
export TRAIN_STEPS="${TRAIN_STEPS:-15000}"
export BATCH_SIZE="${BATCH_SIZE:-192}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"
export SWA_FRACTION="${SWA_FRACTION:-0.20}"
export VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-250}"
export MIN_SELECTION_GREEDY_TOP1="${MIN_SELECTION_GREEDY_TOP1:-0.20}"
export EARLY_STOP_SELECTION_GUARD_FAILURES="${EARLY_STOP_SELECTION_GUARD_FAILURES:-3}"
export EARLY_STOP_AFTER_STEP="${EARLY_STOP_AFTER_STEP:-2000}"
export RAYON_THREADS="${RAYON_THREADS:-32}"

# Namespaced artifact paths so CBDDB runs never collide with AAAAA history.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-cascadiav3/checkpoints/full_v3_${PROFILE}}"
export REPORT="${REPORT:-cascadiav3/reports/full_v3_${PROFILE}_train.json}"
export METRICS="${METRICS:-cascadiav3/reports/full_v3_${PROFILE}_metrics.jsonl}"
export RUNBOOK_REPORT="${RUNBOOK_REPORT:-cascadiav3/reports/full_v3_${PROFILE}_runbook.json}"

exec bash "$(dirname "$0")/run_full_v3_training_pipeline.sh" "$@"
