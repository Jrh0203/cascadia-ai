#!/usr/bin/env bash
set -euo pipefail

# Fresh same-revision exact-final-personal-turn ablation. Both arms are
# regenerated from this checkout; old corrected-rules baselines are not reused
# because their source revision predates the exact-mode implementation.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027071400}"
GAMES="${GAMES:-100}"
JOBS="${JOBS:-12}"
MARKET_DECISION_SAMPLES="${MARKET_DECISION_SAMPLES:-8}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
TAG="exact_k1_20260709_n256_d4"
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
test -s "$MANIFEST"
grep -q 'gumbel-exact-endgame-turns' cascadiav3/real-root-exporter/src/main.rs
if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
  echo "[exact-k1] SOURCE_REVISION does not match HEAD" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[exact-k1] refusing to run from a modified tracked worktree" >&2
  exit 1
fi

install_pinned_zig() {
  if [ -x "$ZIG_ROOT/zig" ]; then
    return
  fi
  local tmp
  tmp="$(mktemp -d)"
  echo "[exact-k1] installing checksum-pinned Zig $ZIG_VERSION under $ZIG_ROOT"
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
  cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
fi

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

report_matches() {
  local report="$1"
  local exact_turns="$2"
  [ -s "$report" ] && "$PYTHON" - "$report" "$RULESET_ID" "$SOURCE_REVISION" \
    "$FIRST_SEED" "$GAMES" "$exact_turns" "$MARKET_DECISION_SAMPLES" <<'PY'
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
    and search.get("determinizations") == 4
    and search.get("market_decision_samples") == int(sys.argv[7])
    and search.get("blend_weight") == 0.5
    and search.get("k_interior") == 16
    and search.get("exact_endgame_turns") == int(sys.argv[6])
    else 1
)
PY
}

run_arm() {
  local arm="$1"
  local exact_turns="$2"
  local games="${3:-$GAMES}"
  local jobs="${4:-$JOBS}"
  local arm_tag="${TAG}_${arm}"
  if [ "$games" != "$GAMES" ]; then
    arm_tag="${arm_tag}_smoke"
  fi
  local report="$REPORT_DIR/${arm_tag}.json"
  if [ "$games" = "$GAMES" ] && report_matches "$report" "$exact_turns"; then
    echo "[exact-k1] reuse $report"
    return
  fi
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
    --gumbel-determinizations 4 \
    --gumbel-market-decision-samples "$MARKET_DECISION_SAMPLES" \
    --gumbel-exact-endgame-turns "$exact_turns" \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "$arm_tag" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${arm_tag}_decisions.jsonl" \
    --summary-out "$REPORT_DIR/${arm_tag}.md"
}

echo "[exact-k1] source=$SOURCE_REVISION seeds=${FIRST_SEED}x${GAMES}"
run_arm exact 1 1 1
run_arm baseline 0
run_arm exact 1

"$PYTHON" -m cascadiav3.compare_exact_endgame \
  --baseline "$REPORT_DIR/${TAG}_baseline.json" \
  --exact "$REPORT_DIR/${TAG}_exact.json" \
  --baseline-decisions "$REPORT_DIR/${TAG}_baseline_decisions.jsonl" \
  --exact-decisions "$REPORT_DIR/${TAG}_exact_decisions.jsonl" \
  --source-revision "$SOURCE_REVISION" \
  --out "$REPORT_DIR/${TAG}_verdict.json" \
  --summary-out "$REPORT_DIR/${TAG}_verdict.md"

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
            "market_decision_samples": int("$MARKET_DECISION_SAMPLES"),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo "[exact-k1] complete"
