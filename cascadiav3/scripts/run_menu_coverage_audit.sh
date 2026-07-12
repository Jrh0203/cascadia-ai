#!/usr/bin/env bash
set -euo pipefail

# R1.3a menu-coverage audit: does the greedy-256 root-menu cap drop the best
# action? Replays the champion cycle4 n1024/d16 ledger twice through the
# exporter's --puzzle-bank mode at n1024/d8 — arm "capped" with the default
# greedy-ranked 256-action root menu, arm "full" with --gumbel-root-menu 0
# (the FULL legal set) — on the same stride-43 roots (~186), then joins the
# arms by (seed, ply) via cascadiav3.analyze_menu_coverage and reports how
# often the full-menu argmax action is absent from the capped menu and the
# full-run Q regret when it is. Full menus are memory-heavy, hence jobs 8.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-/home/john0/venvs/torch/bin/python3}"
LEDGER="${LEDGER:-cascadiav3/reports/rules_20260709_cycle4_n1024_d16_decisions.jsonl}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
CAPPED_DIR="${CAPPED_DIR:-cascadiav3/reports/menu_coverage_20260712_capped}"
FULL_DIR="${FULL_DIR:-cascadiav3/reports/menu_coverage_20260712_full}"
OUT_BASE="${OUT_BASE:-cascadiav3/reports/menu_coverage_20260712}"
STRIDE="${STRIDE:-43}"
JOBS="${JOBS:-8}"

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
    echo "[menu-coverage] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "ledger missing or empty: $LEDGER" test -s "$LEDGER"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
preflight "binary lacks --puzzle-bank (stale build?)" \
  bash -c "\"$BINARY\" --help | grep -q 'puzzle-bank'"
preflight "binary lacks --gumbel-root-menu (stale build?)" \
  bash -c "\"$BINARY\" --help | grep -q 'gumbel-root-menu'"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[menu-coverage] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[menu-coverage] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

MODEL_SERVICE="$BRIDGE_PYTHON -m cascadiav3.torch_inference_bridge --manifest $MANIFEST --device cuda --q-risk-mode mean --policy-mode logits --pairwise-policy-top-k 16"

run_arm() {
  local label="$1" out_dir="$2"
  shift 2
  if [ -e "$out_dir/puzzle_bank_manifest.json" ]; then
    echo "[menu-coverage] reuse existing $label arm $out_dir"
    return 0
  fi
  echo "[menu-coverage] $(date '+%F %T') resolving $label arm (stride=$STRIDE jobs=$JOBS n1024/d8)"
  "$BINARY" \
    --puzzle-bank \
    --in "$LEDGER" \
    --output-dir "$out_dir" \
    --model-service "$MODEL_SERVICE" \
    --model-manifest "$MANIFEST" \
    --model-timeout-ms 600000 \
    --source-revision "$SOURCE_REVISION" \
    --gumbel-n-simulations 1024 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 8 \
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
    "$@"
}

run_arm capped "$CAPPED_DIR"
run_arm full "$FULL_DIR" --gumbel-root-menu 0

echo "[menu-coverage] joining arms"
"$PYTHON" -m cascadiav3.analyze_menu_coverage \
  --capped-dir "$CAPPED_DIR" \
  --full-dir "$FULL_DIR" \
  --out "${OUT_BASE}_analysis.json" \
  --summary-out "${OUT_BASE}_analysis.md"

echo "[menu-coverage] complete: ${OUT_BASE}_analysis.md"
