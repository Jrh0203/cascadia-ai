#!/usr/bin/env bash
set -euo pipefail

# R2.1 puzzle-bank generation (preregistered 2026-07-11, EXPERIMENT_LOG
# 19:40): resolve ~700 stride-selected roots from the champion cycle4
# n1024/d16 ledger with mega-budget search (n4096/top16/d16, repeats 2,
# averaged), worker-pooled at jobs12 against one shared bridge (saturation
# rule). The frozen bank turns future serving candidates into ~45-minute
# regret screens via the same exporter mode at candidate flags + repeats 1,
# scored by cascadiav3.analyze_puzzle_screen. Screens rank candidates for
# gates; they are never promotion evidence.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-/home/john0/venvs/torch/bin/python3}"
LEDGER="${LEDGER:-cascadiav3/reports/rules_20260709_cycle4_n1024_d16_decisions.jsonl}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
BANK_DIR="${BANK_DIR:-cascadiav3/reports/puzzle_bank_20260711_n4096}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
STRIDE="${STRIDE:-11}"
REPEATS="${REPEATS:-2}"
JOBS="${JOBS:-12}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
export CASCADIA_BRIDGE_TF32=0

cd "$ROOT"
mkdir -p "$LOG_DIR"
preflight() {
  local label="$1"
  shift
  if ! "$@"; then
    echo "[puzzle-bank] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "ledger missing or empty: $LEDGER" test -s "$LEDGER"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
preflight "binary lacks --puzzle-bank (stale build?)" \
  bash -c "\"$BINARY\" --help | grep -q 'puzzle-bank'"
if [ -e "$BANK_DIR/puzzle_bank_manifest.json" ]; then
  echo "[puzzle-bank] reuse existing bank $BANK_DIR"
  exit 0
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[puzzle-bank] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[puzzle-bank] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

echo "[puzzle-bank] $(date '+%F %T') generating (stride=$STRIDE repeats=$REPEATS jobs=$JOBS n4096/d16)"
"$BINARY" \
  --puzzle-bank \
  --in "$LEDGER" \
  --output-dir "$BANK_DIR" \
  --model-service "$BRIDGE_PYTHON -m cascadiav3.torch_inference_bridge --manifest $MANIFEST --device cuda --q-risk-mode mean --policy-mode logits --pairwise-policy-top-k 16" \
  --model-manifest "$MANIFEST" \
  --model-timeout-ms 600000 \
  --source-revision "$SOURCE_REVISION" \
  --gumbel-n-simulations 4096 \
  --gumbel-top-m 16 \
  --gumbel-depth-rounds 1 \
  --gumbel-determinizations 16 \
  --gumbel-market-decision-samples 8 \
  --gumbel-exact-endgame-turns 0 \
  --gumbel-blend-weight 0.5 \
  --max-actions 64 \
  --rollout-top-k 4 \
  --k-interior 16 \
  --probe-stride "$STRIDE" \
  --probe-repeats "$REPEATS" \
  --model-sessions "$JOBS" \
  --shared-model-session

echo "[puzzle-bank] complete: $BANK_DIR"
