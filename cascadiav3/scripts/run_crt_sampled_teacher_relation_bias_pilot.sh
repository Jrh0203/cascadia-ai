#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_sampled_teacher_train.jsonl}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_sampled_teacher_train_manifest.json}"
export VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_sampled_teacher_val.jsonl}"
export VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_sampled_teacher_val_manifest.json}"

export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026070000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-100}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026079000}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-25}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-4}"
export MAX_ACTIONS="${MAX_ACTIONS:-16}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-4}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export REGENERATE_ROOTS="${REGENERATE_ROOTS:-1}"

export STEPS="${STEPS:-1600}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LR="${LR:-0.0005}"
export HIDDEN_DIM="${HIDDEN_DIM:-160}"
export LAYERS="${LAYERS:-3}"
export HEADS="${HEADS:-5}"
export MLP_DIM="${MLP_DIM:-320}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-crt-sampled-teacher-relation-bias-v1}"

export REPORT="${REPORT:-cascadiav3/reports/crt_sampled_teacher_relation_bias_pilot.json}"
export CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_sampled_teacher_relation_bias_pilot.pt}"

exec "$SCRIPT_DIR/run_crt_relation_bias_pilot.sh"
