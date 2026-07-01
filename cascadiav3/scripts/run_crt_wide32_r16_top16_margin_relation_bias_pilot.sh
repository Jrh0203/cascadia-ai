#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_train.jsonl}"
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_train_manifest.json}"
export VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_val.jsonl}"
export VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_val_manifest.json}"
export REGENERATE_ROOTS="${REGENERATE_ROOTS:-0}"

export STEPS="${STEPS:-6200}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export LR="${LR:-0.00025}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export LAYERS="${LAYERS:-4}"
export HEADS="${HEADS:-8}"
export MLP_DIM="${MLP_DIM:-512}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16-top16-margin-relation-bias-v1}"

export LOSS_MODE="${LOSS_MODE:-top16-prefilter}"
export Q_LOSS_WEIGHT="${Q_LOSS_WEIGHT:-0.20}"
export POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-0.45}"
export BEST_MARGIN_LOSS_WEIGHT="${BEST_MARGIN_LOSS_WEIGHT:-1.25}"
export PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.35}"
export POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.45}"

export REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16_top16_margin_relation_bias_pilot.json}"
export CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16_top16_margin_relation_bias_pilot.pt}"

"$SCRIPT_DIR/run_crt_relation_bias_pilot.sh"

REPORT="cascadiav3/reports/crt_wide32_r16_top16_margin_prefilter_eval.json" \
PER_ROOT_OUT="cascadiav3/reports/crt_wide32_r16_top16_margin_prefilter_eval_roots.jsonl" \
CHECKPOINT="$CHECKPOINT" \
VAL="$VAL_OUT" \
EXPERIMENT_ID="crt-wide32-r16-top16-margin-prefilter-eval-v1" \
"$SCRIPT_DIR/run_crt_wide32_prefilter_eval.sh"
