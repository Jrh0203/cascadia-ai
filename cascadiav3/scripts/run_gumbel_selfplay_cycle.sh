#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Gumbel self-play expert-iteration cycle (EI-2+).
#
# All four seats are played by the incumbent checkpoint through Gumbel search
# with batched model leaf values over determinized hidden states. Every
# visited root becomes a v2 training record: completed-Q targets, improved
# policy soft targets, and real terminal-outcome value labels. The trainer
# runs the gumbel-selfplay objective with an example-passes cap.
#
# Required:
#   MODEL_MANIFEST  incumbent checkpoint manifest (also drives the bridge)
# The model service defaults to the torch inference bridge on cuda.

export JOB_SLUG="${JOB_SLUG:-gumbel_selfplay_cycle}"
export PROFILE="${PROFILE:-gumbel_selfplay_cycle}"
export EXPERT_TENSOR_MODE="gumbel_selfplay"
export OBJECTIVE="${OBJECTIVE:-gumbel-selfplay}"
export MAX_EXAMPLE_PASSES="${MAX_EXAMPLE_PASSES:-4}"

export MODEL_MANIFEST="${MODEL_MANIFEST:?set MODEL_MANIFEST to the incumbent checkpoint manifest}"
export MODEL_SERVICE="${MODEL_SERVICE:-python -m cascadiav3.torch_inference_bridge --manifest $MODEL_MANIFEST --device cuda}"
export MODEL_TIMEOUT_MS="${MODEL_TIMEOUT_MS:-120000}"
export ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
export MODEL_SESSIONS="${MODEL_SESSIONS:-4}"

# Search shape for data generation (exploration is always on in this mode).
export GUMBEL_N_SIMULATIONS="${GUMBEL_N_SIMULATIONS:-64}"
export GUMBEL_TOP_M="${GUMBEL_TOP_M:-16}"
export GUMBEL_DEPTH_ROUNDS="${GUMBEL_DEPTH_ROUNDS:-1}"
export GUMBEL_DETERMINIZATIONS="${GUMBEL_DETERMINIZATIONS:-4}"
export GUMBEL_BLEND_WEIGHT="${GUMBEL_BLEND_WEIGHT:-0.5}"
export GUMBEL_K_INTERIOR="${GUMBEL_K_INTERIOR:-16}"

# Corpus scale: ~1,250 games x ~80 plies ~= 100k roots per cycle (5x EI-0/1).
export TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026710000}"
export TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-1250}"
export VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026810000}"
export VAL_SEED_COUNT="${VAL_SEED_COUNT:-125}"
export PLIES_PER_SEED="${PLIES_PER_SEED:-80}"

# MAX_ACTIONS bounds the blended-rollout policy inside simulations; root
# menus are always the full legal set.
export MAX_ACTIONS="${MAX_ACTIONS:-8}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
export FILTER_TOP_K="${FILTER_TOP_K:-64}"
export FILTER_MODE="${FILTER_MODE:-top-q-with-selected}"

export MODEL_SIZE="${MODEL_SIZE:-S}"
export TRAIN_STEPS="${TRAIN_STEPS:-25000}"
export BATCH_SIZE="${BATCH_SIZE:-192}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export LR="${LR:-0.0001}"
export SELECTION_METRIC="${SELECTION_METRIC:-locked_val_final_q_regret}"
export SELECTION_MODE="${SELECTION_MODE:-min}"
export INIT_MANIFEST="${INIT_MANIFEST:-$MODEL_MANIFEST}"
export RAYON_THREADS="${RAYON_THREADS:-16}"

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-cascadiav3/checkpoints/full_v3_${PROFILE}}"
export REPORT="${REPORT:-cascadiav3/reports/full_v3_${PROFILE}_train.json}"
export METRICS="${METRICS:-cascadiav3/reports/full_v3_${PROFILE}_metrics.jsonl}"
export RUNBOOK_REPORT="${RUNBOOK_REPORT:-cascadiav3/reports/full_v3_${PROFILE}_runbook.json}"

exec bash "$(dirname "$0")/run_full_v3_training_pipeline.sh" "${1:-status}"
