#!/usr/bin/env bash
set -euo pipefail

# X1 — stronger-teacher distillation (EXPERIMENT_LOG 2026-07-22 01:30
# draft; launched at the 2026-07-22 17:33 milestone-gate failure of the
# from-scratch line under John's autonomy grant).
#
# Shape: the AAAAA champion (zero-shot CBDDB 99.4675 @ n256/d4, 101.2 @
# n1024/d16) generates a CBDDB corpus at n512/d8 — DEEPER search than
# the n256/d4 the student will be evaluated at, so the labels carry
# information the student does not already have (the teacher-student
# gap, inverted). The champion is then fine-tuned on those labels with
# the trust-region anchor (the one warm-start shape never killed).
#
# Required env: SOURCE_REVISION.
# Optional env: TRAIN_FIRST_SEED (2027199000), TRAIN_SEEDS (300),
#   VAL_FIRST_SEED (2027199400), VAL_SEEDS (30), KL_WEIGHT (2.0),
#   L2_WEIGHT (2.0), MAX_PASSES (8), JOBS (24), RAYON (28),
#   EVAL_JOBS (6), N1024_THRESHOLD (100.5).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
TEACHER="cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2027199000}"
TRAIN_SEEDS="${TRAIN_SEEDS:-300}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:-2027199400}"
VAL_SEEDS="${VAL_SEEDS:-30}"
KL_WEIGHT="${KL_WEIGHT:-2.0}"
L2_WEIGHT="${L2_WEIGHT:-2.0}"
MAX_PASSES="${MAX_PASSES:-8}"
JOBS="${JOBS:-24}"
RAYON="${RAYON:-28}"
EVAL_JOBS="${EVAL_JOBS:-6}"
N1024_THRESHOLD="${N1024_THRESHOLD:-100.5}"
BINARY="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter"
PYTHON="${PYTHON:-python3}"
RULESET_ID="cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19"
REPORT_DIR="cascadiav3/reports"
LOG_DIR="cascadiav3/logs"
FIX="cascadiav3/fixtures"
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
hb(){ echo "[$(date "+%F %T")] [cbddb-x1] $*"; }

grep -q 'rules_2026_07_19' cascadiav3/real-root-exporter/src/main.rs
test -s "$TEACHER"

hb "start rev=$SOURCE_REVISION teacher=$TEACHER seeds=${TRAIN_FIRST_SEED}x${TRAIN_SEEDS}+${VAL_FIRST_SEED}x${VAL_SEEDS} kl=$KL_WEIGHT l2=$L2_WEIGHT"

cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

gen_corpus() {
  local tag="$1"; local first="$2"; local count="$3"
  local out="$FIX/cbddb_x1_${tag}_tensor.npz"
  if [ -s "$out" ] && [ -s "$FIX/cbddb_x1_${tag}_manifest.json" ]; then
    hb "GEN $tag reuse $out"
    return
  fi
  hb "GEN $tag starting (${count} seeds @ ${first}, TEACHER n512/d8)"
  "$BINARY" \
    --gumbel-selfplay-tensor-corpus \
    --scoring-cards cbddb \
    --model-service "/home/john0/venvs/torch/bin/python3 -m cascadiav3.torch_inference_bridge --manifest $TEACHER --device cuda" \
    --model-manifest "$TEACHER" \
    --model-timeout-ms 300000 \
    --gumbel-n-simulations 512 --gumbel-top-m 16 --gumbel-depth-rounds 1 \
    --gumbel-determinizations 8 --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 0 --gumbel-blend-weight 0.5 --k-interior 16 \
    --source-revision "$SOURCE_REVISION" \
    --first-seed "$first" --seed-count "$count" --plies-per-seed 80 \
    --max-actions 8 --rollouts-per-action 1 --rollout-top-k 4 \
    --tensor-compression stored \
    --rayon-threads "$RAYON" --model-sessions "$JOBS" --shared-model-session \
    --decisions-out "$FIX/cbddb_x1_${tag}_decisions.jsonl" \
    --out "$out" \
    --manifest "$FIX/cbddb_x1_${tag}_manifest.json"
  hb "GEN $tag DONE"
}

gen_corpus train "$TRAIN_FIRST_SEED" "$TRAIN_SEEDS"
gen_corpus val "$VAL_FIRST_SEED" "$VAL_SEEDS"

hb "TRAIN starting (distill champion on teacher labels, anchor kl=$KL_WEIGHT l2=$L2_WEIGHT)"
if python3 -m cascadiav3.torch_train_cascadiaformer \
  --model-size M \
  --train "$FIX/cbddb_x1_train_tensor.npz" \
  --val "$FIX/cbddb_x1_val_tensor.npz" \
  --train-format npz --val-format npz \
  --steps 2500 --batch-size 192 --grad-accum 1 \
  --lr 0.0001 --weight-decay 0.05 --warmup-fraction 0.02 \
  --device cuda --seed 20260630 \
  --objective gumbel-selfplay \
  --max-example-passes "$MAX_PASSES" \
  --q-quantiles 8 --init-skip-mismatched \
  --selection-metric locked_val_final_q_regret --selection-mode min \
  --val-max-batches 8 --eval-every-steps 250 \
  --swa-fraction 0.20 \
  --init-manifest "$TEACHER" \
  --anchor-manifest "$TEACHER" \
  --anchor-policy-kl-weight "$KL_WEIGHT" \
  --anchor-value-l2-weight "$L2_WEIGHT" \
  --data-workers 0 --tf32 --fused-optimizer --cgab-fused \
  --checkpoint-dir "cascadiav3/checkpoints/cbddb_x1_distill" \
  --metrics-jsonl "$REPORT_DIR/cbddb_x1_distill_metrics.jsonl" \
  --out "$REPORT_DIR/cbddb_x1_distill_train.json" \
  >> "$LOG_DIR/cbddb_x1_train.log" 2>&1; then
  hb "TRAIN COMPLETE"
else
  hb "TRAIN FAILED"; exit 1
fi

X1_MANIFEST="cascadiav3/checkpoints/cbddb_x1_distill/best_locked_val.manifest.json"
test -s "$X1_MANIFEST"

run_eval() {
  local tag="$1"; local simulations="$2"; local determinizations="$3"; local games="$4"
  local report="$REPORT_DIR/cbddb_x1_${tag}.json"
  if [ -s "$report" ]; then hb "EVAL $tag reuse"; return; fi
  hb "EVAL $tag starting (n${simulations}/d${determinizations} x ${games})"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$X1_MANIFEST" \
    --scoring-cards cbddb \
    --device cuda \
    --first-seed "$EVAL_FIRST_SEED" \
    --games "$games" \
    --jobs "$EVAL_JOBS" \
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
    --experiment-id "cbddb_x1_${tag}" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/cbddb_x1_${tag}_decisions.jsonl" \
    --games-out "$REPORT_DIR/cbddb_x1_${tag}_games.jsonl" \
    --summary-out "$REPORT_DIR/cbddb_x1_${tag}.md"
  hb "EVAL $tag DONE"
}

run_eval screen_n256_d4 256 4 100

MEAN=$("$PYTHON" -c "
import json
r = json.load(open('$REPORT_DIR/cbddb_x1_screen_n256_d4.json'))
print(r['strategies']['gumbel-search']['mean_seat_score'])
")
hb "SCREEN mean_seat_score=$MEAN (bar 99.4675; n1024 escalation threshold $N1024_THRESHOLD)"
if "$PYTHON" -c "import sys; sys.exit(0 if float('$MEAN') >= float('$N1024_THRESHOLD') else 1)"; then
  run_eval full_n1024_d16 1024 16 30
else
  hb "EVAL full_n1024_d16 SKIPPED (screen $MEAN < $N1024_THRESHOLD)"
fi

hb "X1 COMPLETE"
