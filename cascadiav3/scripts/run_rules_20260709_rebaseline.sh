#!/usr/bin/env bash
set -euo pipefail

# Fresh post-rules-correction baseline battery. This script is intentionally
# idempotent: a completed report is reused only when its ruleset and exact
# deployed source revision match this launch.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed Git revision}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027070900}"
GAMES="${GAMES:-100}"
JOBS="${JOBS:-12}"
MARKET_DECISION_SAMPLES="${MARKET_DECISION_SAMPLES:-8}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
CYCLE4_MANIFEST="${CYCLE4_MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
DISTQ_MANIFEST="${DISTQ_MANIFEST:-cascadiav3/checkpoints/full_v3_distq_k8/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
ZIG_VERSION="0.13.0"
ZIG_ARCHIVE="zig-linux-x86_64-${ZIG_VERSION}.tar.xz"
ZIG_SHA256="d45312e61ebcc48032b77bc4cf7fd6915c11fa16e4aad116b66c9468211230ea"
ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_ARCHIVE}"
ZIG_ROOT="${ZIG_ROOT:-$HOME/.local/opt/zig-linux-x86_64-${ZIG_VERSION}}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

grep -q 'rules_2026_07_09' cascadiav3/real-root-exporter/src/main.rs
test -s "$CYCLE4_MANIFEST"
test -s "$DISTQ_MANIFEST"

echo "[rebaseline] source_revision=$SOURCE_REVISION ruleset=$RULESET_ID seeds=${FIRST_SEED}x${GAMES}"

install_pinned_zig() {
  if [ -x "$ZIG_ROOT/zig" ]; then
    return
  fi
  local tmp
  tmp="$(mktemp -d)"
  echo "[rebaseline] installing checksum-pinned Zig $ZIG_VERSION under $ZIG_ROOT"
  curl --fail --location --retry 3 --output "$tmp/$ZIG_ARCHIVE" "$ZIG_URL"
  printf '%s  %s\n' "$ZIG_SHA256" "$tmp/$ZIG_ARCHIVE" | sha256sum --check --status
  tar -xJf "$tmp/$ZIG_ARCHIVE" -C "$tmp"
  mkdir -p "$(dirname "$ZIG_ROOT")"
  rm -rf "$ZIG_ROOT"
  mv "$tmp/zig-linux-x86_64-${ZIG_VERSION}" "$ZIG_ROOT"
  rm -rf "$tmp"
}

if command -v cc >/dev/null 2>&1; then
  cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
else
  install_pinned_zig
  export ZIG="$ZIG_ROOT/zig"
  export CC="$ROOT/cascadiav3/scripts/zig-cc-linker.sh"
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC"
  echo "[rebaseline] native cc unavailable; building with $ZIG cc"
  cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
fi

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
  local report="$REPORT_DIR/rules_20260709_cycle4_no_search.json"
  if report_matches "$report"; then
    echo "[rebaseline] reuse $report"
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
    --experiment-id rules-20260709-cycle4-no-search \
    --out "$report" \
    --decisions-out "$REPORT_DIR/rules_20260709_cycle4_no_search_decisions.jsonl" \
    --game-results-out "$REPORT_DIR/rules_20260709_cycle4_no_search_games.jsonl" \
    --summary-out "$REPORT_DIR/rules_20260709_cycle4_no_search.md"
}

run_gumbel() {
  local model="$1"
  local manifest="$2"
  local simulations="$3"
  local determinizations="$4"
  local tag="rules_20260709_${model}_n${simulations}_d${determinizations}"
  local report="$REPORT_DIR/${tag}.json"
  if report_matches "$report"; then
    echo "[rebaseline] reuse $report"
    return
  fi
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$manifest" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --jobs "$JOBS" \
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
    --summary-out "$REPORT_DIR/${tag}.md"
}

# Cheap full-stack smoke before the promotion-scale batteries.
GAMES=1 JOBS=1 run_gumbel cycle4_smoke "$CYCLE4_MANIFEST" 16 2

# Establish the corrected greedy/no-search floor, then make the clean scalar
# versus distributional-Q comparison at both serving budgets on identical
# fresh seeds.
run_no_search
run_gumbel cycle4 "$CYCLE4_MANIFEST" 256 4
run_gumbel distq_k8 "$DISTQ_MANIFEST" 256 4
run_gumbel cycle4 "$CYCLE4_MANIFEST" 1024 16
run_gumbel distq_k8 "$DISTQ_MANIFEST" 1024 16

"$PYTHON" -m cascadiav3.compare_rules_rebaseline \
  --report-dir "$REPORT_DIR" \
  --source-revision "$SOURCE_REVISION" \
  --out "$REPORT_DIR/rules_20260709_rebaseline_verdict.json" \
  --summary-out "$REPORT_DIR/rules_20260709_rebaseline_verdict.md"

"$PYTHON" - "$LOG_DIR/rules_20260709_rebaseline_complete.json" <<PY
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

echo "[rebaseline] complete"
