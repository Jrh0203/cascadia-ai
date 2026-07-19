#!/usr/bin/env bash
set -euo pipefail

# CBDDB campaign cycle runner. One cycle = self-play corpus from the
# current CBDDB incumbent -> warm-start fine-tune -> paired screening
# evals on the fixed screening block (2027190000..99). Authorization:
# EXPERIMENT_LOG 2026-07-19 15:20 ("continue trying to break past 105").
#
# Required env:
#   SOURCE_REVISION  deployed git revision
#   CYCLE_TAG        e.g. c2, c3 (artifact namespace cbddb_<tag>_*)
#   INCUMBENT        manifest of the model that generates the corpus
#                    and warm-starts the fine-tune
#   TRAIN_FIRST_SEED / VAL_FIRST_SEED  fresh, never-used generation
#                    seed blocks (log-audited before launch)
# Optional env: TRAIN_SEEDS (360), VAL_SEEDS (40), STEPS (2500),
#   EVAL_N1024_GAMES (30).
#
# The certification battery (fresh block 2027195000+, 100g n1024/d16)
# is deliberately NOT part of this script — claiming >105 uses
# run_cbddb_certification.sh on a John-visible decision.

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
CYCLE_TAG="${CYCLE_TAG:?set CYCLE_TAG (e.g. c2)}"
INCUMBENT="${INCUMBENT:?set INCUMBENT manifest path}"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:?set TRAIN_FIRST_SEED}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:?set VAL_FIRST_SEED}"
TRAIN_SEEDS="${TRAIN_SEEDS:-360}"
VAL_SEEDS="${VAL_SEEDS:-40}"
STEPS="${STEPS:-2500}"
EVAL_N1024_GAMES="${EVAL_N1024_GAMES:-30}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-12}"
RULESET_ID="cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
FIX="${FIX:-cascadiav3/fixtures}"
EVAL_FIRST_SEED=2027190000

export PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"
if [ -x "$HOME/.local/bin/zig-cc" ] && ! command -v cc >/dev/null 2>&1; then
  export BLAKE3_NO_ASM=1 CC="$HOME/.local/bin/zig-cc"
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$HOME/.local/bin/zig-cc"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED=1
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR" "$FIX"

hb(){ echo "[$(date "+%F %T")] [cbddb-$CYCLE_TAG] $*"; }

grep -q 'rules_2026_07_19' cascadiav3/real-root-exporter/src/main.rs
test -s "$INCUMBENT"

hb "start rev=$SOURCE_REVISION incumbent=$INCUMBENT seeds=${TRAIN_FIRST_SEED}x${TRAIN_SEEDS}+${VAL_FIRST_SEED}x${VAL_SEEDS}"

cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

gen_corpus() {
  local tag="$1"; local first="$2"; local count="$3"
  local out="$FIX/cbddb_${CYCLE_TAG}_${tag}_tensor.npz"
  if [ -s "$out" ] && [ -s "$FIX/cbddb_${CYCLE_TAG}_${tag}_manifest.json" ]; then
    hb "GEN $tag reuse $out"
    return
  fi
  hb "GEN $tag starting (${count} seeds @ ${first}, n256/d4)"
  "$BINARY" \
    --gumbel-selfplay-tensor-corpus \
    --scoring-cards cbddb \
    --model-service "/home/john0/venvs/torch/bin/python3 -m cascadiav3.torch_inference_bridge --manifest $INCUMBENT --device cuda" \
    --model-manifest "$INCUMBENT" \
    --model-timeout-ms 300000 \
    --gumbel-n-simulations 256 --gumbel-top-m 16 --gumbel-depth-rounds 1 \
    --gumbel-determinizations 4 --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 0 --gumbel-blend-weight 0.5 --k-interior 16 \
    --source-revision "$SOURCE_REVISION" \
    --first-seed "$first" --seed-count "$count" --plies-per-seed 80 \
    --max-actions 8 --rollouts-per-action 1 --rollout-top-k 4 \
    --tensor-compression stored \
    --rayon-threads 16 --model-sessions "$JOBS" --shared-model-session \
    --decisions-out "$FIX/cbddb_${CYCLE_TAG}_${tag}_decisions.jsonl" \
    --out "$out" \
    --manifest "$FIX/cbddb_${CYCLE_TAG}_${tag}_manifest.json"
  hb "GEN $tag DONE"
}

gen_corpus train "$TRAIN_FIRST_SEED" "$TRAIN_SEEDS"
gen_corpus val "$VAL_FIRST_SEED" "$VAL_SEEDS"

hb "TRAIN starting"
if python3 -m cascadiav3.torch_train_cascadiaformer \
  --model-size M \
  --train "$FIX/cbddb_${CYCLE_TAG}_train_tensor.npz" \
  --val "$FIX/cbddb_${CYCLE_TAG}_val_tensor.npz" \
  --train-format npz --val-format npz \
  --steps "$STEPS" --batch-size 192 --grad-accum 1 \
  --lr 0.0001 --weight-decay 0.05 --warmup-fraction 0.02 \
  --device cuda --seed 20260630 \
  --objective gumbel-selfplay \
  --max-example-passes 4 \
  --q-quantiles 8 --init-skip-mismatched \
  --selection-metric locked_val_final_q_regret --selection-mode min \
  --val-max-batches 8 --eval-every-steps 250 \
  --swa-fraction 0.20 \
  --init-manifest "$INCUMBENT" \
  --data-workers 4 --prefetch-factor 4 --tf32 --fused-optimizer --cgab-fused \
  --checkpoint-dir "cascadiav3/checkpoints/cbddb_${CYCLE_TAG}_ft" \
  --metrics-jsonl "$REPORT_DIR/cbddb_${CYCLE_TAG}_ft_metrics.jsonl" \
  --out "$REPORT_DIR/cbddb_${CYCLE_TAG}_ft_train.json" \
  >> "$LOG_DIR/cbddb_${CYCLE_TAG}_train.log" 2>&1; then
  hb "TRAIN COMPLETE"
else
  hb "TRAIN FAILED"; exit 1
fi

FT_MANIFEST="cascadiav3/checkpoints/cbddb_${CYCLE_TAG}_ft/best_locked_val.manifest.json"
test -s "$FT_MANIFEST"

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

run_eval() {
  local tag="$1"; local simulations="$2"; local determinizations="$3"; local games="$4"
  local report="$REPORT_DIR/cbddb_${CYCLE_TAG}_${tag}.json"
  if report_matches "$report"; then
    hb "EVAL $tag reuse"
    return
  fi
  hb "EVAL $tag starting (n${simulations}/d${determinizations} x ${games})"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$FT_MANIFEST" \
    --scoring-cards cbddb \
    --device cuda \
    --first-seed "$EVAL_FIRST_SEED" \
    --games "$games" \
    --jobs "$JOBS" \
    --batch-runner \
    --gumbel-n-simulations "$simulations" \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$determinizations" \
    --gumbel-market-decision-samples 8 \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "cbddb_${CYCLE_TAG}_${tag}" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/cbddb_${CYCLE_TAG}_${tag}_decisions.jsonl" \
    --games-out "$REPORT_DIR/cbddb_${CYCLE_TAG}_${tag}_games.jsonl" \
    --summary-out "$REPORT_DIR/cbddb_${CYCLE_TAG}_${tag}.md"
  hb "EVAL $tag DONE"
}

run_eval screen_n256_d4 256 4 100
run_eval screen_n1024_d16 1024 16 "$EVAL_N1024_GAMES"

hb "CYCLE $CYCLE_TAG COMPLETE"
