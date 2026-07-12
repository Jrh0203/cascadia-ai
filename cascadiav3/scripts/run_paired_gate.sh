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
# Group-sequential mode (opt-in; preregistered methodology, EXPERIMENT_LOG
# 2026-07-12):
#   LOOKS           [""] cumulative pair counts, e.g. "40 60 80 100"; the
#                   last look must equal GAMES. When set, arms run in
#                   alternating per-look chunks; after each look the
#                   cumulative merged reports get a sequential verdict
#                   (Lan-DeMets alpha spending, repeated CIs) and the gate
#                   stops as soon as the verdict category is decided.
#   SEQ_RULE        [superiority] or noninferiority
#   SEQ_MARGIN      [-0.25] noninferiority margin (SEQ_RULE=noninferiority)
#   SEQ_ALPHA       [0.05]
#   SEQ_SPENDING    [obrien_fleming] or pocock
#
# Wall-matched gates: set CAND_N so the candidate's expected wall matches
# the baseline (from screen timing), and preregister a wall-parity bound —
# the verdict's timing block records both arms' mean decision seconds.
# Verdicts are promotion evidence only at >=100 seeds (planned final look
# for sequential gates) with a registered block and a preregistered rule;
# the script enforces neither rule — the preregistration entry does.

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
LOOKS="${LOOKS:-}"
SEQ_RULE="${SEQ_RULE:-superiority}"
SEQ_MARGIN="${SEQ_MARGIN:--0.25}"
SEQ_ALPHA="${SEQ_ALPHA:-0.05}"
SEQ_SPENDING="${SEQ_SPENDING:-obrien_fleming}"
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
  local first_seed="$4"
  local games="$5"
  local out_prefix="$6"
  shift 6
  local report="$REPORT_DIR/${out_prefix}.json"
  if [ -s "$report" ]; then
    echo "[paired-gate:$GATE_NAME] reuse $report"
    return
  fi
  hold_gate
  echo "[paired-gate:$GATE_NAME] $(date '+%F %T') arm $arm (n=$n_sims det=$det seeds=${first_seed}x${games})"
  # shellcheck disable=SC2086
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$MANIFEST" \
    --device cuda \
    --first-seed "$first_seed" \
    --games "$games" \
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
    --experiment-id "${out_prefix}" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/${out_prefix}_decisions.jsonl" \
    --games-out "$REPORT_DIR/${out_prefix}_games.jsonl" \
    --summary-out "$REPORT_DIR/${out_prefix}.md" \
    $BASE_FLAGS "$@"
}

VARIED_ARGS=()
for key in $VARIED_KEYS; do
  VARIED_ARGS+=(--varied-key "$key")
done

if [ -z "$LOOKS" ]; then
  echo "[paired-gate:$GATE_NAME] source=$SOURCE_REVISION seeds=${FIRST_SEED}x${GAMES} varied=[$VARIED_KEYS]"
  run_arm baseline "$BASE_N" "$BASE_DET" "$FIRST_SEED" "$GAMES" "${TAG}_baseline"
  # shellcheck disable=SC2086
  run_arm candidate "$CAND_N" "$CAND_DET" "$FIRST_SEED" "$GAMES" "${TAG}_candidate" $CAND_FLAGS

  "$PYTHON" -m cascadiav3.compare_search_shape \
    --baseline "$REPORT_DIR/${TAG}_baseline.json" \
    --candidate "$REPORT_DIR/${TAG}_candidate.json" \
    --source-revision "$SOURCE_REVISION" \
    "${VARIED_ARGS[@]}" \
    --out "$REPORT_DIR/${TAG}_verdict.json" \
    --summary-out "$REPORT_DIR/${TAG}_verdict.md" >/dev/null

  echo "[paired-gate:$GATE_NAME] complete: $REPORT_DIR/${TAG}_verdict.md"
  exit 0
fi

# --- Group-sequential mode (preregistered methodology) ---------------------
LAST_LOOK="${LOOKS##* }"
if [ "$LAST_LOOK" != "$GAMES" ]; then
  echo "[paired-gate:$GATE_NAME] LOOKS must end at GAMES ($LAST_LOOK != $GAMES)" >&2
  exit 1
fi
LOOKS_CSV="${LOOKS// /,}"
echo "[paired-gate:$GATE_NAME] SEQUENTIAL source=$SOURCE_REVISION seeds=${FIRST_SEED}x${GAMES} looks=[$LOOKS_CSV] rule=$SEQ_RULE varied=[$VARIED_KEYS]"

merge_arm() {
  local arm="$1"
  shift
  local merge_args=()
  local look
  for look in "$@"; do
    merge_args+=(--chunk "$REPORT_DIR/${TAG}_${arm}_c${look}.json")
    merge_args+=(--decisions "$REPORT_DIR/${TAG}_${arm}_c${look}_decisions.jsonl")
  done
  "$PYTHON" -m cascadiav3.merge_benchmark_reports \
    "${merge_args[@]}" \
    --experiment-id "${TAG}_${arm}" \
    --out "$REPORT_DIR/${TAG}_${arm}.json" >/dev/null
}

prev=0
completed_looks=()
decision="continue"
for look in $LOOKS; do
  chunk=$((look - prev))
  chunk_first=$((FIRST_SEED + prev))
  run_arm baseline "$BASE_N" "$BASE_DET" "$chunk_first" "$chunk" "${TAG}_baseline_c${look}"
  # shellcheck disable=SC2086
  run_arm candidate "$CAND_N" "$CAND_DET" "$chunk_first" "$chunk" "${TAG}_candidate_c${look}" $CAND_FLAGS
  completed_looks+=("$look")
  merge_arm baseline "${completed_looks[@]}"
  merge_arm candidate "${completed_looks[@]}"
  decision=$("$PYTHON" -m cascadiav3.sequential_gate \
    --baseline "$REPORT_DIR/${TAG}_baseline.json" \
    --candidate "$REPORT_DIR/${TAG}_candidate.json" \
    --source-revision "$SOURCE_REVISION" \
    "${VARIED_ARGS[@]}" \
    --looks "$LOOKS_CSV" \
    --alpha "$SEQ_ALPHA" \
    --spending "$SEQ_SPENDING" \
    --rule "$SEQ_RULE" \
    --margin "$SEQ_MARGIN" \
    --out "$REPORT_DIR/${TAG}_look${look}_verdict.json" \
    --summary-out "$REPORT_DIR/${TAG}_look${look}_verdict.md" | tail -1)
  cp "$REPORT_DIR/${TAG}_look${look}_verdict.json" "$REPORT_DIR/${TAG}_verdict.json"
  cp "$REPORT_DIR/${TAG}_look${look}_verdict.md" "$REPORT_DIR/${TAG}_verdict.md"
  echo "[paired-gate:$GATE_NAME] $(date '+%F %T') look at $look pairs: $decision"
  prev=$look
  if [ "$decision" != "continue" ]; then
    break
  fi
done

echo "[paired-gate:$GATE_NAME] complete (sequential, $decision at $prev pairs): $REPORT_DIR/${TAG}_verdict.md"
