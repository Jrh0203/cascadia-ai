#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_scaled_sampled_teacher_train.jsonl}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_scaled_sampled_teacher_train_manifest.json}"
export VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_scaled_sampled_teacher_val.jsonl}"
export VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_scaled_sampled_teacher_val_manifest.json}"

export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026080000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-400}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026089000}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-100}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-4}"
export MAX_ACTIONS="${MAX_ACTIONS:-16}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-8}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export REGENERATE_ROOTS="${REGENERATE_ROOTS:-1}"

export STEPS="${STEPS:-3200}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LR="${LR:-0.0004}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export LAYERS="${LAYERS:-4}"
export HEADS="${HEADS:-8}"
export MLP_DIM="${MLP_DIM:-512}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-crt-scaled-sampled-teacher-relation-bias-v1}"

export REPORT="${REPORT:-cascadiav3/reports/crt_scaled_sampled_teacher_relation_bias_pilot.json}"
export CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_scaled_sampled_teacher_relation_bias_pilot.pt}"

exec "$SCRIPT_DIR/run_crt_relation_bias_pilot.sh"
