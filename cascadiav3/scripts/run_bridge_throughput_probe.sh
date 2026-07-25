#!/usr/bin/env bash
export PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"
set -euo pipefail

# Bridge serving-path throughput probe: eager vs opt-in fast paths.
# It may run on any host and alongside other work.

ROOT="${ROOT:-/home/john0/cascadia}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
BATCH_SIZES="${BATCH_SIZES:-8,32,96,192}"
ARMS="${ARMS:-eager,bucket,compile,compile_bucket}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-20}"
ROOTS="${ROOTS:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The probe module toggles COMPILE/BUCKET per arm itself.
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
unset CASCADIA_BRIDGE_COMPILE
unset CASCADIA_BRIDGE_BUCKET

cd "$ROOT"
mkdir -p "$REPORT_DIR"
test -s "$MANIFEST"

if [ -z "$ROOTS" ]; then
  # CPU-only dry-run root export at serving-realistic menu sizes; identical
  # recipe to run_model_throughput_probe.sh but with fuller menus.
  test -x "$BINARY"
  "$BINARY" \
    --chance-mcts-dry-run \
    --allow-model-fallback \
    --first-seed 2027071600 \
    --seed-count 2 \
    --plies-per-seed 8 \
    --max-actions 256 \
    --rollouts-per-action 1 \
    --rollout-top-k 1 \
    --rollout-determinize \
    --out "$TMP/roots.jsonl" \
    --manifest "$TMP/roots.manifest.json"
  ROOTS="$TMP/roots.jsonl"
fi
test -s "$ROOTS"

"$PYTHON" -m cascadiav3.torch_bridge_throughput_probe \
  --manifest "$MANIFEST" \
  --roots "$ROOTS" \
  --batch-sizes "$BATCH_SIZES" \
  --arms "$ARMS" \
  --warmup-iterations "$WARMUP" \
  --measured-iterations "$ITERS" \
  --device "$DEVICE" \
  --out "$REPORT_DIR/bridge_throughput_probe.json" \
  --summary-out "$REPORT_DIR/bridge_throughput_probe.md"

echo "[bridge-throughput] complete: $REPORT_DIR/bridge_throughput_probe.{json,md}"
