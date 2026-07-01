#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_train.jsonl}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_train_manifest.json}"
export VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_val.jsonl}"
export VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_val_manifest.json}"

export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026100000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-300}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026109000}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-75}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-4}"
export MAX_ACTIONS="${MAX_ACTIONS:-32}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-16}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export REGENERATE_ROOTS="${REGENERATE_ROOTS:-1}"

export STEPS="${STEPS:-5200}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export LR="${LR:-0.00035}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export LAYERS="${LAYERS:-4}"
export HEADS="${HEADS:-8}"
export MLP_DIM="${MLP_DIM:-512}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16-sampled-teacher-relation-bias-v1}"

export LOSS_MODE="${LOSS_MODE:-standard}"
export REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16_sampled_teacher_relation_bias_pilot.json}"
export CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16_sampled_teacher_relation_bias_pilot.pt}"

"$SCRIPT_DIR/run_crt_relation_bias_pilot.sh"

REPORT="cascadiav3/reports/crt_wide32_r16_prefilter_eval.json" \
PER_ROOT_OUT="cascadiav3/reports/crt_wide32_r16_prefilter_eval_roots.jsonl" \
CHECKPOINT="$CHECKPOINT" \
VAL="$VAL_OUT" \
EXPERIMENT_ID="crt-wide32-r16-prefilter-eval-v1" \
"$SCRIPT_DIR/run_crt_wide32_prefilter_eval.sh"
