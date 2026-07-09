#!/usr/bin/env bash
set -euo pipefail

# Engineering-only CUDA shared-bridge concurrency calibration. Runs matched
# jobs12/16/24 arms, records one-second GPU telemetry, and recommends the
# smallest arm within 2% of the fastest only when throughput improves >=5%.
# It never mutates serving/generation defaults automatically.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
FIRST_SEED="${FIRST_SEED:-2027073400}"
GAMES="${GAMES:-48}"
RULESET_ID="cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09"
MANIFEST="${MANIFEST:-cascadiav3/checkpoints/full_v3_distq_k8/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
TAG="cuda_concurrency_20260709_n64_d4"
ZIG_VERSION="0.13.0"
ZIG_ARCHIVE="zig-linux-x86_64-${ZIG_VERSION}.tar.xz"
ZIG_SHA256="d45312e61ebcc48032b77bc4cf7fd6915c11fa16e4aad116b66c9468211230ea"
ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_ARCHIVE}"
ZIG_ROOT="${ZIG_ROOT:-$HOME/.local/opt/zig-linux-x86_64-${ZIG_VERSION}}"
PROFILE_PID=""

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED="${CASCADIA_CGAB_FUSED:-1}"
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"
export CASCADIA_BRIDGE_TF32=0
unset CASCADIA_BRIDGE_AUTOCAST
unset CASCADIA_BRIDGE_BUCKET
unset CASCADIA_BRIDGE_COMPILE

stop_profile() {
  if [ -n "$PROFILE_PID" ]; then
    kill "$PROFILE_PID" 2>/dev/null || true
    wait "$PROFILE_PID" 2>/dev/null || true
    PROFILE_PID=""
  fi
}
trap stop_profile EXIT

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"
test -s "$MANIFEST"
test "$GAMES" -ge 24
command -v nvidia-smi >/dev/null
grep -q 'dynamic_seed_queue' cascadiav3/src/cascadiav3/torch_cascadiaformer_gumbel_benchmark.py
test -s cascadiav3/src/cascadiav3/compare_cuda_concurrency.py
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[cuda-concurrency] SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[cuda-concurrency] refusing to run from a modified tracked worktree" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[cuda-concurrency] source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

install_pinned_zig() {
  if [ -x "$ZIG_ROOT/zig" ]; then
    return
  fi
  local tmp
  tmp="$(mktemp -d)"
  echo "[cuda-concurrency] installing checksum-pinned Zig $ZIG_VERSION under $ZIG_ROOT"
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
test -x "$BINARY"

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

report_matches() {
  local jobs="$1"
  local prefix="$REPORT_DIR/${TAG}_jobs${jobs}"
  local report="${prefix}.json"
  local decisions="${prefix}_decisions.jsonl"
  local games="${prefix}_games.jsonl"
  local profile="${prefix}_gpu.csv"
  [ -s "$report" ] \
    && [ -s "$decisions" ] \
    && [ -s "$games" ] \
    && [ -s "$profile" ] \
    && [ "$(wc -l < "$decisions" | tr -d '[:space:]')" = "$((GAMES * 80))" ] \
    && [ "$(wc -l < "$games" | tr -d '[:space:]')" = "$GAMES" ] \
    && [ "$(wc -l < "$profile" | tr -d '[:space:]')" -ge 30 ] \
    && "$PYTHON" - "$report" "$RULESET_ID" "$SOURCE_REVISION" \
      "$FIRST_SEED" "$GAMES" "$jobs" "$MANIFEST" "$BINARY" <<'PY'
import json
import sys
from pathlib import Path

from cascadiav3.torch_cascadiaformer_gumbel_benchmark import model_artifact_provenance

report = json.load(open(sys.argv[1], encoding="utf-8"))
expected_seeds = list(range(int(sys.argv[4]), int(sys.argv[4]) + int(sys.argv[5])))
expected_jobs = int(sys.argv[6])
expected_artifacts = model_artifact_provenance(Path(sys.argv[8]), Path(sys.argv[7]))
artifact_keys = (
    "binary_sha256",
    "manifest_sha256",
    "weights_sha256",
    "checkpoint_tag",
    "checkpoint_step",
    "q_quantiles",
)
search = report.get("search", {})
execution = report.get("execution", {})
artifacts = report.get("artifacts", {})
raise SystemExit(
    0
    if report.get("status") == "pass"
    and report.get("scientific_eligibility") == "candidate_only_search_arm"
    and report.get("ruleset_id") == sys.argv[2]
    and report.get("source_revision") == sys.argv[3]
    and report.get("seeds") == expected_seeds
    and report.get("manifest") == sys.argv[7]
    and all(artifacts.get(key) == expected_artifacts.get(key) for key in artifact_keys)
    and report.get("control", {}).get("kind") == "none"
    and search.get("n_simulations") == 64
    and search.get("top_m") == 16
    and search.get("depth_rounds") == 1
    and search.get("determinizations") == 4
    and search.get("market_decision_samples") == 8
    and search.get("exact_endgame_turns") == 0
    and search.get("blend_weight") == 0.5
    and search.get("parallel_leaf_rollouts") is False
    and search.get("k_interior") == 16
    and search.get("q_risk_mode") == "mean"
    and search.get("policy_mode") == "logits"
    and execution.get("runner") == "gumbel-benchmark-batch"
    and execution.get("batch_runner") is True
    and execution.get("requested_jobs") == expected_jobs
    and execution.get("seed_count") == len(expected_seeds)
    and execution.get("parallel_game_cap") == min(expected_jobs, len(expected_seeds))
    and execution.get("seed_scheduler") == "dynamic_seed_queue"
    and execution.get("shared_model_session") is True
    and execution.get("bridge_process_topology") == "one_shared_bridge"
    and execution.get("maximum_concurrent_bridge_processes") == 1
    and execution.get("device") == "cuda"
    else 1
)
PY
}

start_profile() {
  local path="$1"
  rm -f "$path" "${path}.stderr"
  nvidia-smi \
    --query-gpu=utilization.gpu,power.draw,memory.used,temperature.gpu \
    --format=csv,noheader,nounits \
    -l 1 > "$path" 2> "${path}.stderr" &
  PROFILE_PID=$!
}

run_arm() {
  local jobs="$1"
  local prefix="$REPORT_DIR/${TAG}_jobs${jobs}"
  if report_matches "$jobs"; then
    echo "[cuda-concurrency] reuse ${prefix}.json"
    return
  fi
  rm -f "${prefix}.json" "${prefix}.md" "${prefix}_decisions.jsonl" \
    "${prefix}_games.jsonl" "${prefix}_gpu.csv" "${prefix}_gpu.csv.stderr"
  start_profile "${prefix}_gpu.csv"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$MANIFEST" \
    --device cuda \
    --first-seed "$FIRST_SEED" \
    --games "$GAMES" \
    --jobs "$jobs" \
    --batch-runner \
    --gumbel-n-simulations 64 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 4 \
    --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 0 \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "${TAG}_jobs${jobs}" \
    --out "${prefix}.json" \
    --decisions-out "${prefix}_decisions.jsonl" \
    --games-out "${prefix}_games.jsonl" \
    --summary-out "${prefix}.md"
  stop_profile
  report_matches "$jobs"
}

echo "[cuda-concurrency] source=$SOURCE_REVISION seeds=${FIRST_SEED}x${GAMES}"
run_arm 12
run_arm 16
run_arm 24

"$PYTHON" -m cascadiav3.compare_cuda_concurrency \
  --jobs12-report "$REPORT_DIR/${TAG}_jobs12.json" \
  --jobs12-decisions "$REPORT_DIR/${TAG}_jobs12_decisions.jsonl" \
  --jobs12-games "$REPORT_DIR/${TAG}_jobs12_games.jsonl" \
  --jobs12-gpu-profile "$REPORT_DIR/${TAG}_jobs12_gpu.csv" \
  --jobs16-report "$REPORT_DIR/${TAG}_jobs16.json" \
  --jobs16-decisions "$REPORT_DIR/${TAG}_jobs16_decisions.jsonl" \
  --jobs16-games "$REPORT_DIR/${TAG}_jobs16_games.jsonl" \
  --jobs16-gpu-profile "$REPORT_DIR/${TAG}_jobs16_gpu.csv" \
  --jobs24-report "$REPORT_DIR/${TAG}_jobs24.json" \
  --jobs24-decisions "$REPORT_DIR/${TAG}_jobs24_decisions.jsonl" \
  --jobs24-games "$REPORT_DIR/${TAG}_jobs24_games.jsonl" \
  --jobs24-gpu-profile "$REPORT_DIR/${TAG}_jobs24_gpu.csv" \
  --source-revision "$SOURCE_REVISION" \
  --out "$REPORT_DIR/${TAG}_verdict.json" \
  --summary-out "$REPORT_DIR/${TAG}_verdict.md"

"$PYTHON" - "$REPORT_DIR/${TAG}_verdict.json" "$LOG_DIR/${TAG}_complete.json" <<'PY'
import json
import sys

verdict = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "status": "complete",
            "source_revision": verdict["source_revision"],
            "games": len(verdict["seeds"]),
            "recommended_jobs": verdict["selection"]["recommended_jobs"],
            "recommendation": verdict["selection"]["recommendation"],
            "verdict": sys.argv[1],
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

echo "[cuda-concurrency] complete; no runtime default changed"
