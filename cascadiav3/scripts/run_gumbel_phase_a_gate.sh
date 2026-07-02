#!/usr/bin/env bash
set -euo pipefail

# Phase A gate: 100 paired games, Gumbel search (serving budget) versus the
# HONEST full rollout-search control (--rollout-determinize on, so the control
# no longer peeks at the true hidden tile/bag order). Promotion requires the
# paired delta's 95% CI to exclude zero in the candidate's favor.
#
# Run on the box that owns the GPU (john0). Requires MANIFEST.

MANIFEST="${MANIFEST:?set MANIFEST to the checkpoint manifest}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
DEVICE="${DEVICE:-cuda}"
GAMES="${GAMES:-100}"
FIRST_SEED="${FIRST_SEED:-2026995000}"
JOBS="${JOBS:-4}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
  --binary "$BINARY" \
  --manifest "$MANIFEST" \
  --device "$DEVICE" \
  --first-seed "$FIRST_SEED" \
  --games "$GAMES" \
  --jobs "$JOBS" \
  --gumbel-n-simulations "${GUMBEL_N_SIMULATIONS:-64}" \
  --gumbel-top-m "${GUMBEL_TOP_M:-16}" \
  --gumbel-depth-rounds "${GUMBEL_DEPTH_ROUNDS:-1}" \
  --gumbel-determinizations "${GUMBEL_DETERMINIZATIONS:-4}" \
  --gumbel-blend-weight "${GUMBEL_BLEND_WEIGHT:-0.5}" \
  --k-interior "${GUMBEL_K_INTERIOR:-16}" \
  --control full-search \
  --control-max-actions "${CONTROL_MAX_ACTIONS:-64}" \
  --control-rollouts-per-action "${CONTROL_ROLLOUTS_PER_ACTION:-16}" \
  --control-rollout-top-k "${CONTROL_ROLLOUT_TOP_K:-4}" \
  --control-workers "${CONTROL_WORKERS:-4}" \
  --model-timeout-ms "${MODEL_TIMEOUT_MS:-300000}" \
  --experiment-id "${EXPERIMENT_ID:-gumbel-phase-a-gate-v1}" \
  --out "${OUT:-cascadiav3/reports/gumbel_phase_a_gate.json}" \
  --summary-out "${SUMMARY_OUT:-cascadiav3/reports/gumbel_phase_a_gate_summary.md}"
