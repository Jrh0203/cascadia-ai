#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Phase B ceiling probe: lavish Gumbel search budget, pure value bootstrap
# (w=1.0), full legal root menus, no timing gate. Answers whether the current
# value model can reach the 100-point band when search cost is no object:
#   mean >= 100 -> search-budget/distillation problem;
#   97-100      -> model-bound but close (2-3 EI cycles projected);
#   < 97        -> model-bound; prioritize data scale and value quality.
#
# Run on the box that owns the GPU (john0). Requires MANIFEST.

MANIFEST="${MANIFEST:?set MANIFEST to the checkpoint manifest}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
DEVICE="${DEVICE:-cuda}"
GAMES="${GAMES:-20}"
FIRST_SEED="${FIRST_SEED:-2026995000}"
JOBS="${JOBS:-4}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
  --binary "$BINARY" \
  --manifest "$MANIFEST" \
  --device "$DEVICE" \
  --first-seed "$FIRST_SEED" \
  --games "$GAMES" \
  --jobs "$JOBS" \
  --gumbel-n-simulations "${GUMBEL_N_SIMULATIONS:-512}" \
  --gumbel-top-m "${GUMBEL_TOP_M:-32}" \
  --gumbel-depth-rounds "${GUMBEL_DEPTH_ROUNDS:-1}" \
  --gumbel-determinizations "${GUMBEL_DETERMINIZATIONS:-8}" \
  --gumbel-blend-weight "${GUMBEL_BLEND_WEIGHT:-1.0}" \
  --k-interior "${GUMBEL_K_INTERIOR:-16}" \
  --control "${CONTROL:-full-search}" \
  --control-max-actions "${CONTROL_MAX_ACTIONS:-64}" \
  --control-rollouts-per-action "${CONTROL_ROLLOUTS_PER_ACTION:-16}" \
  --control-rollout-top-k "${CONTROL_ROLLOUT_TOP_K:-4}" \
  --control-workers "${CONTROL_WORKERS:-4}" \
  --model-timeout-ms "${MODEL_TIMEOUT_MS:-300000}" \
  --experiment-id "${EXPERIMENT_ID:-gumbel-ceiling-probe-v1}" \
  --out "${OUT:-cascadiav3/reports/gumbel_ceiling_probe.json}" \
  --summary-out "${SUMMARY_OUT:-cascadiav3/reports/gumbel_ceiling_probe_summary.md}"
