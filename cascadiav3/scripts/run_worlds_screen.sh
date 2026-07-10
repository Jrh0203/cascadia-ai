#!/usr/bin/env bash
set -euo pipefail

# Worlds-allocation screen (preregistered 2026-07-10, EXPERIMENT_LOG 11:20):
# cycle4 scalar, n256/top16, K1 on, samples 8, det4 vs det8, fresh seed block
# 2027071500..1599. Determinizations cycle inside the fixed simulation
# budget, so both arms are equal-cost. The paired verdict runs orchestrator-
# side via cascadiav3.compare_search_shape (trace identity is NOT expected —
# world count changes evaluations at every ply).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027071500}"
GAMES="${GAMES:-100}"
JOBS="${JOBS:-12}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
TAG="worlds_screen_20260710_n256"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
test -s "$MANIFEST"
test -x "$BINARY"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[worlds-screen] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[worlds-screen] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

hold_gate() {
  while [ -e "$LOG_DIR/HOLD_worlds_screen" ]; do
    echo "[worlds-screen] $(date '+%F %T') holding on HOLD_worlds_screen"
    sleep 60
  done
}

report_matches() {
  local report="$1"
  local worlds="$2"
  [ -s "$report" ] && "$PYTHON" - "$report" "$RULESET_ID" "$SOURCE_REVISION" \
    "$FIRST_SEED" "$GAMES" "$worlds" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected_seeds = list(range(int(sys.argv[4]), int(sys.argv[4]) + int(sys.argv[5])))
search = report.get("search", {})
raise SystemExit(
    0
    if report.get("status") == "pass"
    and report.get("ruleset_id") == sys.argv[2]
    and report.get("source_revision") == sys.argv[3]
    and report.get("seeds") == expected_seeds
    and report.get("control", {}).get("kind") == "none"
    and search.get("n_simulations") == 256
    and search.get("top_m") == 16
    and search.get("depth_rounds") == 1
    and search.get("determinizations") == int(sys.argv[6])
    and search.get("market_decision_samples") == 8
    and search.get("blend_weight") == 0.5
    and search.get("k_interior") == 16
    and search.get("exact_endgame_turns") == 1
    else 1
)
PY
}

run_arm() {
  local worlds="$1"
  local games="${2:-$GAMES}"
  local jobs="${3:-$JOBS}"
  local arm_tag="${TAG}_det${worlds}"
  if [ "$games" != "$GAMES" ]; then
    arm_tag="${arm_tag}_smoke"
  fi
  local report="$REPORT_DIR/${arm_tag}.json"
  if [ "$games" = "$GAMES" ] && report_matches "$report" "$worlds"; then
    echo "[worlds-screen] reuse $report"
    return
  fi
  hold_gate
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$MANIFEST" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$games" \
    --jobs "$jobs" \
    --batch-runner \
    --gumbel-n-simulations 256 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$worlds" \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 1 \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "$arm_tag" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${arm_tag}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${arm_tag}_games.jsonl" \
    --summary-out "$REPORT_DIR/${arm_tag}.md"
}

echo "[worlds-screen] source=$SOURCE_REVISION seeds=${FIRST_SEED}x${GAMES} arms=det4,det8"
run_arm 8 1 1
run_arm 4
run_arm 8

"$PYTHON" - "$LOG_DIR/${TAG}_complete.json" <<PY
import json
import sys

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": "complete",
            "ruleset_id": "$RULESET_ID",
            "source_revision": "$SOURCE_REVISION",
            "first_seed": int("$FIRST_SEED"),
            "games": int("$GAMES"),
            "arms": ["det4", "det8"],
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo "[worlds-screen] complete"
