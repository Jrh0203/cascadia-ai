#!/usr/bin/env bash
set -euo pipefail

# EI-0: greedy-state search bootstrap.
#
# This is the first real expert-iteration bootstrap for CascadiaFormer. Roots
# come from greedy self-play states, each retained K32 menu is labeled by
# sampled greedy rollouts, and the actual state trajectory still advances by
# the greedy action to avoid the rollout-state distribution-shift failure.

export JOB_SLUG="${JOB_SLUG:-cascadiaformer_ei0_greedy_search_bootstrap}"
export PROFILE="${PROFILE:-ei0_greedy_search_bootstrap}"
export EXPERT_TENSOR_MODE="${EXPERT_TENSOR_MODE:-greedy_search_bootstrap}"
export MAX_ACTIONS="${MAX_ACTIONS:-32}"
export FILTER_TOP_K="${FILTER_TOP_K:-32}"
export FILTER_MODE="${FILTER_MODE:-greedy-prefix-strict}"
export OBJECTIVE="${OBJECTIVE:-search-improved-greedy-retention}"
export SELECTION_METRIC="${SELECTION_METRIC:-locked_val_total}"
export SELECTION_MODE="${SELECTION_MODE:-min}"
export MODEL_SIZE="${MODEL_SIZE:-S}"
export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026410000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-250}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026510000}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-50}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-80}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export TRAIN_STEPS="${TRAIN_STEPS:-25000}"
export BATCH_SIZE="${BATCH_SIZE:-192}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-0.0001}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"
export VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
export EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-250}"
export MIN_SELECTION_GREEDY_TOP1="${MIN_SELECTION_GREEDY_TOP1:-0.20}"
export EARLY_STOP_SELECTION_GUARD_FAILURES="${EARLY_STOP_SELECTION_GUARD_FAILURES:-3}"
export EARLY_STOP_AFTER_STEP="${EARLY_STOP_AFTER_STEP:-2000}"
export SWA_FRACTION="${SWA_FRACTION:-0.20}"
export RAYON_THREADS="${RAYON_THREADS:-32}"
export INIT_MANIFEST="${INIT_MANIFEST:-cascadiav3/checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-cascadiav3/checkpoints/full_v3_${PROFILE}}"
export REPORT="${REPORT:-cascadiav3/reports/full_v3_${PROFILE}_train.json}"
export METRICS="${METRICS:-cascadiav3/reports/full_v3_${PROFILE}_metrics.jsonl}"
export RUNBOOK_REPORT="${RUNBOOK_REPORT:-cascadiav3/reports/full_v3_${PROFILE}_runbook.json}"

exec bash "$(dirname "$0")/run_full_v3_training_pipeline.sh" "${1:-status}"
