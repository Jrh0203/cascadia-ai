#!/usr/bin/env bash
set -euo pipefail

# R1.1a contention audit (2026-07-10): replay the champion cycle4 n1024/d16
# decisions ledger without search and, for every decision, compare the
# chosen action against the best model-Q alternative under the TABLE
# objective (value-head per-seat sums at each afterstate). Bounds the
# cooperative-play prize before any training. Measurement only — no gate,
# no promotion evidence; MPS execution is acceptable (DEVICE=mps).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-/home/john0/venvs/torch/bin/python3}"
DEVICE="${DEVICE:-cuda}"
LEDGER="${LEDGER:-cascadiav3/reports/rules_20260709_cycle4_n1024_d16_decisions.jsonl}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
TAG="table_contention_audit_20260710"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
export CASCADIA_BRIDGE_TF32=0

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
preflight() {
  local label="$1"
  shift
  if ! "$@"; then
    echo "[contention-audit] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "ledger missing or empty: $LEDGER" test -s "$LEDGER"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
preflight "binary lacks --table-contention-audit (stale build?)" \
  bash -c "\"$BINARY\" --help | grep -q 'table-contention-audit'"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[contention-audit] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[contention-audit] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ] && [ "$DEVICE" = "cuda" ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

AUDIT_OUT="$REPORT_DIR/${TAG}.jsonl"
if [ ! -s "$AUDIT_OUT" ]; then
  echo "[contention-audit] $(date '+%F %T') auditing $LEDGER (device=$DEVICE)"
  "$BINARY" \
    --table-contention-audit \
    --in "$LEDGER" \
    --out "$AUDIT_OUT" \
    --model-service "$BRIDGE_PYTHON -m cascadiav3.torch_inference_bridge --manifest $MANIFEST --device $DEVICE --q-risk-mode mean --policy-mode logits --pairwise-policy-top-k 16" \
    --model-manifest "$MANIFEST" \
    --model-timeout-ms 300000
else
  echo "[contention-audit] reuse $AUDIT_OUT"
fi

"$PYTHON" -m cascadiav3.analyze_table_contention \
  --in "$AUDIT_OUT" \
  --out "$REPORT_DIR/${TAG}_analysis.json" \
  --summary-out "$REPORT_DIR/${TAG}_analysis.md"

echo "[contention-audit] complete: $REPORT_DIR/${TAG}_analysis.md"
