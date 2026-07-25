#!/usr/bin/env bash
set -euo pipefail

# Engineering-only fixed-root throughput probe for the tiny-model / huge-search
# direction. This never emits gameplay strength evidence.

ROOT="${ROOT:-/home/john0/cascadia}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
M_MANIFEST="${M_MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
S_MANIFEST="${S_MANIFEST:-cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json}"
TAG="model_throughput_20260709"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
export CASCADIA_BRIDGE_TF32=0
unset CASCADIA_BRIDGE_AUTOCAST
unset CASCADIA_BRIDGE_BUCKET
unset CASCADIA_BRIDGE_COMPILE

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
test -x "$BINARY"
test -s "$M_MANIFEST"
test -s "$S_MANIFEST"
"$BINARY" \
  --chance-mcts-dry-run \
  --allow-model-fallback \
  --first-seed 2027071600 \
  --seed-count 1 \
  --plies-per-seed 4 \
  --max-actions 64 \
  --rollouts-per-action 1 \
  --rollout-top-k 1 \
  --rollout-determinize \
  --out "$TMP/roots.jsonl" \
  --manifest "$TMP/roots.manifest.json"

test "$(wc -l < "$TMP/roots.jsonl" | tr -d '[:space:]')" = 4

"$PYTHON" -m cascadiav3.torch_model_throughput_benchmark \
  --roots "$TMP/roots.jsonl" \
  --manifest "cycle4_M=$M_MANIFEST" \
  --manifest "ei0_S=$S_MANIFEST" \
  --synthetic-model-sizes XS,tiny \
  --root-format production-packed \
  --batch-sizes 1,2,4,8,16,32 \
  --warmup-iterations 3 \
  --measured-iterations 10 \
  --device "$DEVICE" \
  --baseline-label cycle4_M \
  --out "$REPORT_DIR/${TAG}_${DEVICE}.json" \
  --summary-out "$REPORT_DIR/${TAG}_${DEVICE}.md"

"$PYTHON" - "$LOG_DIR/${TAG}_${DEVICE}_complete.json" <<PY
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": "complete",
            "device": "$DEVICE",
            "report": "$REPORT_DIR/${TAG}_${DEVICE}.json",
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo "[model-throughput] complete"
