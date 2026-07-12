#!/usr/bin/env bash
set -euo pipefail

# Generic R2.1 bank screen: replay the frozen mega-budget puzzle bank's roots
# (default cascadiav3/reports/puzzle_bank_20260711_n4096, stride 11, repeats 2,
# n4096/d16) with a CANDIDATE serving configuration (n256/top16/d4, repeats 1)
# plus any candidate-specific exporter flags, then score the candidate by bank
# regret via cascadiav3.analyze_puzzle_screen. One screen per SCREEN_NAME;
# candidate deltas ride in EXTRA_FLAGS, e.g.
#   SCREEN_NAME=ghost_opp EXTRA_FLAGS="--gumbel-ghost-opponents" \
#   SOURCE_REVISION=$(git rev-parse HEAD) bash cascadiav3/scripts/run_bank_screen.sh
#   SCREEN_NAME=c025_topk8 EXTRA_FLAGS="--gumbel-c-scale 0.25 --gumbel-sigma-norm topk:8" ...
# Screens rank candidates for gates; they are never promotion evidence.

ROOT="${ROOT:-/home/john0/cascadia}"
SCREEN_NAME="${SCREEN_NAME:?set SCREEN_NAME to a short candidate label (used in the output dir)}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
# EXTRA_FLAGS is expanded UNQUOTED into the exporter argv so one string can
# carry several flags (word-splitting is intentional). Flag values containing
# spaces or glob characters are not supported — pass them another way.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-/home/john0/venvs/torch/bin/python3}"
LEDGER="${LEDGER:-cascadiav3/reports/rules_20260709_cycle4_n1024_d16_decisions.jsonl}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
BANK_DIR="${BANK_DIR:-cascadiav3/reports/puzzle_bank_20260711_n4096}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
STRIDE="${STRIDE:-11}"
JOBS="${JOBS:-12}"
SCREEN_DIR="cascadiav3/reports/puzzle_screen_${SCREEN_NAME}"

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
    echo "[bank-screen] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "ledger missing or empty: $LEDGER" test -s "$LEDGER"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
preflight "binary lacks --puzzle-bank (stale build?)" \
  bash -c "\"$BINARY\" --help | grep -q 'puzzle-bank'"
preflight "frozen bank missing: $BANK_DIR/puzzle_bank_manifest.json" \
  test -s "$BANK_DIR/puzzle_bank_manifest.json"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[bank-screen] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[bank-screen] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

MODEL_SERVICE="$BRIDGE_PYTHON -m cascadiav3.torch_inference_bridge --manifest $MANIFEST --device cuda --q-risk-mode mean --policy-mode logits --pairwise-policy-top-k 16"

if [ -e "$SCREEN_DIR/puzzle_bank_manifest.json" ]; then
  echo "[bank-screen] reuse existing screen $SCREEN_DIR"
else
  echo "[bank-screen] $(date '+%F %T') screening '$SCREEN_NAME' (stride=$STRIDE jobs=$JOBS n256/d4 extra: ${EXTRA_FLAGS:-none})"
  # shellcheck disable=SC2086  # EXTRA_FLAGS word-splitting is intentional (documented above)
  "$BINARY" \
    --puzzle-bank \
    --in "$LEDGER" \
    --output-dir "$SCREEN_DIR" \
    --model-service "$MODEL_SERVICE" \
    --model-manifest "$MANIFEST" \
    --model-timeout-ms 600000 \
    --source-revision "$SOURCE_REVISION" \
    --gumbel-n-simulations 256 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 4 \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 0 \
    --gumbel-blend-weight 0.5 \
    --max-actions 64 \
    --rollout-top-k 4 \
    --k-interior 16 \
    --probe-stride "$STRIDE" \
    --probe-repeats 1 \
    --model-sessions "$JOBS" \
    --shared-model-session \
    $EXTRA_FLAGS
fi

echo "[bank-screen] scoring against bank $BANK_DIR"
"$PYTHON" -m cascadiav3.analyze_puzzle_screen \
  --bank-dir "$BANK_DIR" \
  --screen-dir "$SCREEN_DIR" \
  --out "${SCREEN_DIR}_analysis.json" \
  --summary-out "${SCREEN_DIR}_analysis.md"

echo "[bank-screen] complete: ${SCREEN_DIR}_analysis.md"
