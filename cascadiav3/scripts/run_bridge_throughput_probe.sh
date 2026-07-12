#!/usr/bin/env bash
export PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"
set -euo pipefail

# Engineering-only bridge serving-path throughput probe (R2.4): eager vs the
# opt-in fast paths (CASCADIA_BRIDGE_COMPILE / _BUCKET), TF32 off, plus a
# numerics check (max abs diff vs eager) that decides whether adoption needs
# a paired score gate. Run ON john0, only when the GPU is idle. This never
# emits gameplay strength evidence.

# --- Refuse to run while serving/search is live (john0 is one-job-at-a-time).
if ps aux | grep -E 'gumbel|torch_inference_bridge' | grep -v grep | grep -q .; then
  echo "==================================================================" >&2
  echo "[bridge-throughput] REFUSING TO START: a gumbel/torch_inference_bridge" >&2
  echo "[bridge-throughput] process is running. john0 runs strictly one job" >&2
  echo "[bridge-throughput] at a time; wait for the current experiment to end." >&2
  echo "==================================================================" >&2
  exit 1
fi
# Belt and braces: refuse if the GPU is measurably busy even without a
# matching process name (WSL2 nvidia-smi does not always list processes).
NVIDIA_SMI="$(command -v nvidia-smi || true)"
if [ -n "$NVIDIA_SMI" ]; then
  GPU_UTIL="$("$NVIDIA_SMI" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || echo "")"
  if [ -n "$GPU_UTIL" ] && [ "$GPU_UTIL" -ge 10 ]; then
    echo "[bridge-throughput] REFUSING TO START: GPU utilization is ${GPU_UTIL}% (>=10%)." >&2
    exit 1
  fi
fi

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}"
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
# Production serving env, minus every numerics-affecting knob: TF32 hard off
# (battery parity), autocast off; the probe module toggles COMPILE/BUCKET
# per arm itself.
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
export CASCADIA_BRIDGE_TF32=0
unset CASCADIA_BRIDGE_AUTOCAST
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
  --source-revision "$SOURCE_REVISION" \
  --out "$REPORT_DIR/bridge_throughput_probe.json" \
  --summary-out "$REPORT_DIR/bridge_throughput_probe.md"

echo "[bridge-throughput] complete: $REPORT_DIR/bridge_throughput_probe.{json,md}"
