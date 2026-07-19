#!/usr/bin/env bash
set -euo pipefail

# CBDDB smoke test, Stage 1: ZERO-SHOT transfer. The AAAAA champion
# (cycle4 scalar, best_locked_val) plays the CBDDB research ruleset
# (Bear C, Elk B, Salmon D, Hawk D, Fox B; Elk-B strict-diamond per
# John's 2026-07-19 ruling) with no retraining. Preregistration:
# EXPERIMENT_LOG 2026-07-19 09:40. Historical old-tech anchors (base):
# greedy-MCE-750 ~96.5, NNUE-MCE-750 ~97.2 (measured under the looser
# legacy Elk-B rule, so slightly generous vs this identity).
#
# Arms: 1-game full-stack smoke, no-search floors, n256/d4 x 100.
# Seeds 2027190000..99, fresh block, touch-once.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed Git revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027190000}"
GAMES="${GAMES:-100}"
JOBS="${JOBS:-12}"
MARKET_DECISION_SAMPLES="${MARKET_DECISION_SAMPLES:-8}"
RULESET_ID="cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19"
CYCLE4_MANIFEST="${CYCLE4_MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"

export PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"
if [ -x "$HOME/.local/bin/zig-cc" ] && ! command -v cc >/dev/null 2>&1; then
  export BLAKE3_NO_ASM=1 CC="$HOME/.local/bin/zig-cc"
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$HOME/.local/bin/zig-cc"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

grep -q 'rules_2026_07_19' cascadiav3/real-root-exporter/src/main.rs
test -s "$CYCLE4_MANIFEST"

echo "[cbddb-s1] source_revision=$SOURCE_REVISION ruleset=$RULESET_ID seeds=${FIRST_SEED}x${GAMES}"

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
  local report="$REPORT_DIR/cbddb_smoke_s1_no_search.json"
  if report_matches "$report"; then
    echo "[cbddb-s1] reuse $report"
    return
  fi
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_game_benchmark \
    --binary "$BINARY" \
    --manifest "$CYCLE4_MANIFEST" \
    --scoring-cards cbddb \
    --selection-heads policy,q \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --max-actions 256 \
    --baseline-workers "$JOBS" \
    --treatment-workers 1 \
    --device cuda \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id cbddb-smoke-s1-no-search \
    --out "$report" \
    --decisions-out "$REPORT_DIR/cbddb_smoke_s1_no_search_decisions.jsonl" \
    --game-results-out "$REPORT_DIR/cbddb_smoke_s1_no_search_games.jsonl" \
    --summary-out "$REPORT_DIR/cbddb_smoke_s1_no_search.md"
}

run_gumbel() {
  local tag_suffix="$1"
  local simulations="$2"
  local determinizations="$3"
  local games="${4:-$GAMES}"
  local jobs="${5:-$JOBS}"
  local tag="cbddb_smoke_s1_${tag_suffix}_n${simulations}_d${determinizations}"
  local report="$REPORT_DIR/${tag}.json"
  if report_matches "$report"; then
    echo "[cbddb-s1] reuse $report"
    return
  fi
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$CYCLE4_MANIFEST" \
    --scoring-cards cbddb \
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

# Cheap full-stack smoke, then the floors, then the zero-shot battery.
run_gumbel smoke 16 2 1 1
run_no_search
run_gumbel zeroshot 256 4

echo "[cbddb-s1] CBDDB STAGE1 COMPLETE"
