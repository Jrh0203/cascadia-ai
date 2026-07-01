#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_train.jsonl}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_train_manifest.json}"
export VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl}"
export VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_val_manifest.json}"

export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026120000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-120}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026129000}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-30}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-20}"
export MAX_ACTIONS="${MAX_ACTIONS:-32}"
export ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-16}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export REGENERATE_ROOTS="${REGENERATE_ROOTS:-1}"

export STEPS="${STEPS:-7600}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export LR="${LR:-0.00032}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export LAYERS="${LAYERS:-4}"
export HEADS="${HEADS:-8}"
export MLP_DIM="${MLP_DIM:-512}"
export TRAIN_MODULE="${TRAIN_MODULE:-cascadiav3.torch_semantic_relation_bias_merit}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p20-semantic-relation-bias-v1}"

export LOSS_MODE="${LOSS_MODE:-standard}"
export REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p20_semantic_relation_bias_pilot.json}"
export CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16p20_semantic_relation_bias_pilot.pt}"

"$SCRIPT_DIR/run_crt_relation_bias_pilot.sh"

REPORT="cascadiav3/reports/crt_wide32_r16p20_semantic_prefilter_eval.json" \
PER_ROOT_OUT="cascadiav3/reports/crt_wide32_r16p20_semantic_prefilter_eval_roots.jsonl" \
CHECKPOINT="$CHECKPOINT" \
VAL="$VAL_OUT" \
EXPERIMENT_ID="crt-wide32-r16p20-semantic-prefilter-eval-v1" \
"$SCRIPT_DIR/run_crt_wide32_prefilter_eval.sh"
