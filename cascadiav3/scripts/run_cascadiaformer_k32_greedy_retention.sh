#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_SLUG="${JOB_SLUG:-cascadiaformer_k32_greedy_retention}"
export PROFILE="${PROFILE:-phase0_bootstrap_jsonl}"
export MAX_ACTIONS="${MAX_ACTIONS:-32}"
export FILTER_TOP_K="${FILTER_TOP_K:-32}"
export FILTER_MODE="${FILTER_MODE:-greedy-prefix-strict}"
export OBJECTIVE="${OBJECTIVE:-pure-greedy-retention}"
export SELECTION_METRIC="${SELECTION_METRIC:-locked_val_greedy_policy}"
export SELECTION_MODE="${SELECTION_MODE:-min}"
export MODEL_SIZE="${MODEL_SIZE:-S}"
export TRAIN_STEPS="${TRAIN_STEPS:-1500}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-0.0005}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_FRACTION="${WARMUP_FRACTION:-0.03}"
export VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
export SWA_FRACTION="${SWA_FRACTION:-0.20}"
export SEED="${SEED:-2026070101}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-cascadiav3/checkpoints/full_v3_${PROFILE}_k32_greedy_retention}"
export REPORT="${REPORT:-cascadiav3/reports/full_v3_${PROFILE}_k32_greedy_retention_train.json}"
export METRICS="${METRICS:-cascadiav3/reports/full_v3_${PROFILE}_k32_greedy_retention_metrics.jsonl}"
export RUNBOOK_REPORT="${RUNBOOK_REPORT:-cascadiav3/reports/full_v3_${PROFILE}_k32_greedy_retention_runbook.json}"

exec bash "$SCRIPT_DIR/run_full_v3_training_pipeline.sh" "${1:-status}"
