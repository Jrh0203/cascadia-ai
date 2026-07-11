#!/usr/bin/env bash
set -euo pipefail

# R0.2 offline kill test (preregistered 2026-07-10): replay the champion
# cycle4 n1024/d16 decisions ledger, sample ~100 real serving roots (stride
# 79 spans all 100 games), and re-run each root search 6x unpaired vs 6x
# paired (CRN) rollouts at the n256/d4 screen tier with matched search seeds.
# Preregistered rule: proceed to the n256 gate iff pooled top1-top2
# completed-Q gap variance (visited) drops >= 20% under pairing.

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
PROBE_STRIDE="${PROBE_STRIDE:-79}"
PROBE_REPEATS="${PROBE_REPEATS:-6}"
PROBE_MAX_ROOTS="${PROBE_MAX_ROOTS:-100}"
TAG="search_stability_probe_20260710"

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
    echo "[stability-probe] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "ledger missing or empty: $LEDGER" test -s "$LEDGER"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
preflight "binary lacks --search-stability-probe (stale build?)" \
  bash -c "\"$BINARY\" --help | grep -q 'search-stability-probe'"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[stability-probe] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[stability-probe] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ] && [ "$DEVICE" = "cuda" ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

PROBE_OUT="$REPORT_DIR/${TAG}.jsonl"
if [ ! -s "$PROBE_OUT" ]; then
  echo "[stability-probe] $(date '+%F %T') probing (stride=$PROBE_STRIDE repeats=$PROBE_REPEATS max_roots=$PROBE_MAX_ROOTS device=$DEVICE)"
  "$BINARY" \
    --search-stability-probe \
    --in "$LEDGER" \
    --out "$PROBE_OUT" \
    --model-service "$BRIDGE_PYTHON -m cascadiav3.torch_inference_bridge --manifest $MANIFEST --device $DEVICE --q-risk-mode mean --policy-mode logits --pairwise-policy-top-k 16" \
    --model-manifest "$MANIFEST" \
    --model-timeout-ms 300000 \
    --gumbel-n-simulations 256 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 4 \
    --gumbel-market-decision-samples 8 \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --probe-stride "$PROBE_STRIDE" \
    --probe-repeats "$PROBE_REPEATS" \
    --probe-max-roots "$PROBE_MAX_ROOTS"
else
  echo "[stability-probe] reuse $PROBE_OUT"
fi

"$PYTHON" -m cascadiav3.analyze_search_stability \
  --in "$PROBE_OUT" \
  --out "$REPORT_DIR/${TAG}_analysis.json" \
  --summary-out "$REPORT_DIR/${TAG}_analysis.md" \
  --variance-reduction-floor 0.20

echo "[stability-probe] complete: $REPORT_DIR/${TAG}_analysis.md"
