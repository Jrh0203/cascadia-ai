#!/usr/bin/env bash
set -euo pipefail

# Generic preregistered paired gate: two benchmark arms (baseline +
# candidate) on ONE fresh seed block, verdict via compare_search_shape.
# Every gate is env-parameterized so preregistered experiments run as
# queue entries with zero new code:
#
#   GATE_NAME       (required) artifact prefix: gate_<GATE_NAME>_*
#   FIRST_SEED      (required) fresh block start — REGISTER IT FIRST
#   GAMES           [100]
#   VARIED_KEYS     (required) space-separated search-provenance keys that
#                   differ between arms (e.g. "ghost_opponents n_simulations")
#   BASE_FLAGS      [""] extra exporter flags for BOTH arms (unquoted expansion)
#   CAND_FLAGS      (required) extra exporter flags for the candidate arm only
#   BASE_N / CAND_N [256/256] per-arm --gumbel-n-simulations
#   BASE_DET / CAND_DET [4/4] per-arm --gumbel-determinizations
#   EXACT_ENDGAME   [1] K1 setting (both arms)
#   JOBS            [12]
#
# Wall-matched gates: set CAND_N so the candidate's expected wall matches
# the baseline (from screen timing), and preregister a wall-parity bound —
# the verdict's timing block records both arms' mean decision seconds.
# Verdicts are promotion evidence only at >=100 seeds with a registered
# block and a preregistered rule; the script enforces neither rule — the
# preregistration entry does.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
GATE_NAME="${GATE_NAME:?set GATE_NAME}"
FIRST_SEED="${FIRST_SEED:?set FIRST_SEED to a registered fresh block}"
VARIED_KEYS="${VARIED_KEYS:?set VARIED_KEYS (space-separated)}"
CAND_FLAGS="${CAND_FLAGS:?set CAND_FLAGS (candidate-arm exporter flags)}"
BASE_FLAGS="${BASE_FLAGS:-}"
GAMES="${GAMES:-100}"
BASE_N="${BASE_N:-256}"
CAND_N="${CAND_N:-256}"
BASE_DET="${BASE_DET:-4}"
CAND_DET="${CAND_DET:-4}"
EXACT_ENDGAME="${EXACT_ENDGAME:-1}"
JOBS="${JOBS:-12}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
TAG="gate_${GATE_NAME}"

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
    echo "[paired-gate:$GATE_NAME] preflight failed: $label" >&2
    exit 1
  fi
}
preflight "manifest missing or empty: $MANIFEST" test -s "$MANIFEST"
preflight "exporter binary missing: $BINARY" test -x "$BINARY"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[paired-gate:$GATE_NAME] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[paired-gate:$GATE_NAME] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

hold_gate() {
  while [ -e "$LOG_DIR/HOLD_paired_gate" ]; do
    echo "[paired-gate:$GATE_NAME] $(date '+%F %T') holding on HOLD_paired_gate"
    sleep 60
  done
}

run_arm() {
  local arm="$1"
  local n_sims="$2"
  local det="$3"
  shift 3
  local report="$REPORT_DIR/${TAG}_${arm}.json"
  if [ -s "$report" ]; then
    echo "[paired-gate:$GATE_NAME] reuse $report"
    return
  fi
  hold_gate
  echo "[paired-gate:$GATE_NAME] $(date '+%F %T') arm $arm (n=$n_sims det=$det seeds=${FIRST_SEED}x${GAMES})"
  # shellcheck disable=SC2086
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$MANIFEST" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --jobs "$JOBS" \
    --batch-runner \
    --gumbel-n-simulations "$n_sims" \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$det" \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns "$EXACT_ENDGAME" \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "${TAG}_${arm}" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${TAG}_${arm}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${TAG}_${arm}_games.jsonl" \
    --summary-out "$REPORT_DIR/${TAG}_${arm}.md" \
    $BASE_FLAGS "$@"
}

echo "[paired-gate:$GATE_NAME] source=$SOURCE_REVISION seeds=${FIRST_SEED}x${GAMES} varied=[$VARIED_KEYS]"
run_arm baseline "$BASE_N" "$BASE_DET"
# shellcheck disable=SC2086
run_arm candidate "$CAND_N" "$CAND_DET" $CAND_FLAGS

VARIED_ARGS=()
for key in $VARIED_KEYS; do
  VARIED_ARGS+=(--varied-key "$key")
done
"$PYTHON" -m cascadiav3.compare_search_shape \
  --baseline "$REPORT_DIR/${TAG}_baseline.json" \
  --candidate "$REPORT_DIR/${TAG}_candidate.json" \
  --source-revision "$SOURCE_REVISION" \
  "${VARIED_ARGS[@]}" \
  --out "$REPORT_DIR/${TAG}_verdict.json" \
  --summary-out "$REPORT_DIR/${TAG}_verdict.md" >/dev/null

echo "[paired-gate:$GATE_NAME] complete: $REPORT_DIR/${TAG}_verdict.md"
