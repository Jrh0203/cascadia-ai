#!/usr/bin/env bash
set -euo pipefail

# Direct baseline/candidate strength comparison.
#
# The historical filename is retained for compatibility. This is not a gate:
# it has no preregistration, seed allocation, source identity, hash checks,
# HOLD files, minimum game count, sequential boundary, or promotion verdict.
# Watch either arm while it runs and change the configuration whenever useful.

ROOT="${ROOT:-/home/john0/cascadia}"
MATCHUP_NAME="${MATCHUP_NAME:-${GATE_NAME:-matchup}}"
FIRST_SEED="${FIRST_SEED:-1}"
GAMES="${GAMES:-32}"
BASE_FLAGS="${BASE_FLAGS:-}"
CAND_FLAGS="${CAND_FLAGS:-}"
BASE_N="${BASE_N:-256}"
CAND_N="${CAND_N:-256}"
BASE_DET="${BASE_DET:-4}"
CAND_DET="${CAND_DET:-4}"
EXACT_ENDGAME="${EXACT_ENDGAME:-1}"
JOBS="${JOBS:-12}"
REUSE="${REUSE:-0}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
BASE_MANIFEST="${BASE_MANIFEST:-${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}}"
CAND_MANIFEST="${CAND_MANIFEST:-$BASE_MANIFEST}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
TAG="matchup_${MATCHUP_NAME}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-cascadiav3/src}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
test -x "$BINARY"
test -s "$BASE_MANIFEST"
test -s "$CAND_MANIFEST"

run_arm() {
  local arm="$1"
  local manifest="$2"
  local n_sims="$3"
  local determinizations="$4"
  local extra_flags="$5"
  local report="$REPORT_DIR/${TAG}_${arm}.json"
  if [ "$REUSE" = "1" ] && [ -s "$report" ]; then
    echo "[matchup:$MATCHUP_NAME] reusing $report"
    return
  fi
  echo "[matchup:$MATCHUP_NAME] $arm: games=$GAMES n=$n_sims det=$determinizations"
  # shellcheck disable=SC2086
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$manifest" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --jobs "$JOBS" \
    --batch-runner \
    --gumbel-n-simulations "$n_sims" \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$determinizations" \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns "$EXACT_ENDGAME" \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --experiment-id "${TAG}_${arm}" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${TAG}_${arm}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${TAG}_${arm}_games.jsonl" \
    --summary-out "$REPORT_DIR/${TAG}_${arm}.md" \
    $BASE_FLAGS $extra_flags
}

run_arm baseline "$BASE_MANIFEST" "$BASE_N" "$BASE_DET" ""
run_arm candidate "$CAND_MANIFEST" "$CAND_N" "$CAND_DET" "$CAND_FLAGS"

"$PYTHON" -m cascadiav3.compare_search_shape \
  --baseline "$REPORT_DIR/${TAG}_baseline.json" \
  --candidate "$REPORT_DIR/${TAG}_candidate.json" \
  --varied-key determinizations \
  --varied-key manifest \
  --out "$REPORT_DIR/${TAG}_comparison.json" \
  --summary-out "$REPORT_DIR/${TAG}_comparison.md"

echo "[matchup:$MATCHUP_NAME] complete: $REPORT_DIR/${TAG}_comparison.md"
