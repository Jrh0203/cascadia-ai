#!/usr/bin/env bash
set -euo pipefail

# Gate 0: fresh canonical champion baseline under the 2026-07-16 rules
# identity (successor minted by the wildlife-bag conservation fix,
# 45fb5072). The 07-09 baselines are a closed evidence boundary; every
# later comparison, gate, or promotion argument under the new rules
# anchors to the numbers this battery produces. Preregistration:
# EXPERIMENT_LOG 2026-07-18. This is a measurement, not a hypothesis
# test — no decision rule is attached, and its decision ledgers double
# as the incumbent-measured game set for M1 tomography replay.
#
# Idempotent like the 07-09 rebaseline: a completed report is reused
# only when its ruleset and exact deployed source revision match.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed Git revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027160000}"
GAMES="${GAMES:-100}"
JOBS="${JOBS:-12}"
MARKET_DECISION_SAMPLES="${MARKET_DECISION_SAMPLES:-8}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16"
CYCLE4_MANIFEST="${CYCLE4_MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

grep -q 'rules_2026_07_16' cascadiav3/real-root-exporter/src/main.rs
test -s "$CYCLE4_MANIFEST"

echo "[gate0] source_revision=$SOURCE_REVISION ruleset=$RULESET_ID seeds=${FIRST_SEED}x${GAMES}"

cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

report_matches() {
  local report="$1"
  [ -s "$report" ] && "$PYTHON" - "$report" "$RULESET_ID" "$SOURCE_REVISION" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(
    0
    if report.get("status") == "pass"
    and report.get("ruleset_id") == sys.argv[2]
    and report.get("source_revision") == sys.argv[3]
    else 1
)
PY
}

run_no_search() {
  local report="$REPORT_DIR/rules_20260716_gate0_cycle4_no_search.json"
  if report_matches "$report"; then
    echo "[gate0] reuse $report"
    return
  fi
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_game_benchmark \
    --binary "$BINARY" \
    --manifest "$CYCLE4_MANIFEST" \
    --selection-heads policy,q \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --max-actions 256 \
    --baseline-workers "$JOBS" \
    --treatment-workers 1 \
    --device cuda \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id rules-20260716-gate0-cycle4-no-search \
    --out "$report" \
    --decisions-out "$REPORT_DIR/rules_20260716_gate0_cycle4_no_search_decisions.jsonl" \
    --game-results-out "$REPORT_DIR/rules_20260716_gate0_cycle4_no_search_games.jsonl" \
    --summary-out "$REPORT_DIR/rules_20260716_gate0_cycle4_no_search.md"
}

run_gumbel() {
  local model="$1"
  local manifest="$2"
  local simulations="$3"
  local determinizations="$4"
  local games="${5:-$GAMES}"
  local jobs="${6:-$JOBS}"
  local tag="rules_20260716_gate0_${model}_n${simulations}_d${determinizations}"
  local report="$REPORT_DIR/${tag}.json"
  if report_matches "$report"; then
    echo "[gate0] reuse $report"
    return
  fi
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$manifest" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$games" \
    --jobs "$jobs" \
    --batch-runner \
    --gumbel-n-simulations "$simulations" \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$determinizations" \
    --gumbel-market-decision-samples "$MARKET_DECISION_SAMPLES" \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "$tag" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${tag}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${tag}_games.jsonl" \
    --summary-out "$REPORT_DIR/${tag}.md"
}

# Cheap full-stack smoke before the promotion-scale batteries.
run_gumbel cycle4_smoke "$CYCLE4_MANIFEST" 16 2 1 1

# Champion identity only (cycle4 scalar): the greedy floor, the cheap
# generation-grade battery, and the canonical n1024/d16 champion arm.
# The distq arms from the 07-09 rebaseline are omitted per the budget
# ruling — Gate 0 is scoped to ~0.4 GPU-day.
run_no_search
run_gumbel cycle4 "$CYCLE4_MANIFEST" 256 4
run_gumbel cycle4 "$CYCLE4_MANIFEST" 1024 16

"$PYTHON" - "$LOG_DIR/rules_20260716_gate0_complete.json" <<PY
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
            "market_decision_samples": int("$MARKET_DECISION_SAMPLES"),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo "[gate0] GATE0 BATTERY COMPLETE"
