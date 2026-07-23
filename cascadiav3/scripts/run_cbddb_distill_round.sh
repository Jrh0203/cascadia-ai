#!/usr/bin/env bash
set -euo pipefail

# CBDDB distillation ladder — one round. Generalization of
# run_cbddb_x1.sh (X1 = round 1 with TEACHER = AAAAA champion).
#
# Shape per round: the current-best model (TEACHER) generates a fresh
# CBDDB corpus at n512/d8 — deeper search than the n256/d4 screen the
# student faces — then the SAME model is fine-tuned on those labels
# with the trust-region anchor. Each round's teacher is genuinely
# stronger than the previous student, so the loop does not eat its own
# tail the way same-budget self-play does.
#
# Per John's ruling 2026-07-23: NO full-battery (n1024/d16) eval inside
# rounds — the paired n256/d4 x100 screen is the go/no-go signal.
# The full battery runs once, manually, when the ladder stalls or
# pre-certification.
#
# Required env: SOURCE_REVISION, ROUND_TAG (e.g. x2), TEACHER (manifest
#   path on john0), TRAIN_FIRST_SEED, VAL_FIRST_SEED (fresh blocks,
#   audited in EXPERIMENT_LOG before launch).
# Optional env: TRAIN_SEEDS (300), VAL_SEEDS (30), KL_WEIGHT (2.0),
#   L2_WEIGHT (2.0), MAX_PASSES (8), JOBS (24), RAYON (28),
#   EVAL_JOBS (6), GEN_SIMS (512), GEN_DETS (8).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
ROUND_TAG="${ROUND_TAG:?set ROUND_TAG (e.g. x2)}"
TEACHER="${TEACHER:?set TEACHER manifest path}"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:?set TRAIN_FIRST_SEED (fresh block)}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:?set VAL_FIRST_SEED (fresh block)}"
TRAIN_SEEDS="${TRAIN_SEEDS:-300}"
VAL_SEEDS="${VAL_SEEDS:-30}"
KL_WEIGHT="${KL_WEIGHT:-2.0}"
L2_WEIGHT="${L2_WEIGHT:-2.0}"
MAX_PASSES="${MAX_PASSES:-8}"
JOBS="${JOBS:-24}"
RAYON="${RAYON:-28}"
EVAL_JOBS="${EVAL_JOBS:-6}"
GEN_SIMS="${GEN_SIMS:-512}"
GEN_DETS="${GEN_DETS:-8}"
BINARY="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter"
PYTHON="${PYTHON:-python3}"
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
hb(){ echo "[$(date "+%F %T")] [cbddb-$ROUND_TAG] $*"; }

grep -q 'rules_2026_07_19' cascadiav3/real-root-exporter/src/main.rs
test -s "$TEACHER"

hb "start rev=$SOURCE_REVISION teacher=$TEACHER seeds=${TRAIN_FIRST_SEED}x${TRAIN_SEEDS}+${VAL_FIRST_SEED}x${VAL_SEEDS} gen=n${GEN_SIMS}/d${GEN_DETS} kl=$KL_WEIGHT l2=$L2_WEIGHT"

cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

gen_corpus() {
  local tag="$1"; local first="$2"; local count="$3"
  local out="$FIX/cbddb_${ROUND_TAG}_${tag}_tensor.npz"
  if [ -s "$out" ] && [ -s "$FIX/cbddb_${ROUND_TAG}_${tag}_manifest.json" ]; then
    hb "GEN $tag reuse $out"
    return
  fi
  hb "GEN $tag starting (${count} seeds @ ${first}, TEACHER n${GEN_SIMS}/d${GEN_DETS})"
  "$BINARY" \
    --gumbel-selfplay-tensor-corpus \
    --scoring-cards cbddb \
    --model-service "/home/john0/venvs/torch/bin/python3 -m cascadiav3.torch_inference_bridge --manifest $TEACHER --device cuda" \
    --model-manifest "$TEACHER" \
    --model-timeout-ms 300000 \
    --gumbel-n-simulations "$GEN_SIMS" --gumbel-top-m 16 --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$GEN_DETS" --gumbel-market-decision-samples 8 \
    --gumbel-exact-endgame-turns 0 --gumbel-blend-weight 0.5 --k-interior 16 \
    --source-revision "$SOURCE_REVISION" \
    --first-seed "$first" --seed-count "$count" --plies-per-seed 80 \
    --max-actions 8 --rollouts-per-action 1 --rollout-top-k 4 \
    --tensor-compression stored \
    --rayon-threads "$RAYON" --model-sessions "$JOBS" --shared-model-session \
    --decisions-out "$FIX/cbddb_${ROUND_TAG}_${tag}_decisions.jsonl" \
    --out "$out" \
    --manifest "$FIX/cbddb_${ROUND_TAG}_${tag}_manifest.json"
  hb "GEN $tag DONE"
}

gen_corpus train "$TRAIN_FIRST_SEED" "$TRAIN_SEEDS"
gen_corpus val "$VAL_FIRST_SEED" "$VAL_SEEDS"

hb "TRAIN starting (anchored distill, kl=$KL_WEIGHT l2=$L2_WEIGHT, passes=$MAX_PASSES)"
if python3 -m cascadiav3.torch_train_cascadiaformer \
  --model-size M \
  --train "$FIX/cbddb_${ROUND_TAG}_train_tensor.npz" \
  --val "$FIX/cbddb_${ROUND_TAG}_val_tensor.npz" \
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
  --checkpoint-dir "cascadiav3/checkpoints/cbddb_${ROUND_TAG}_distill" \
  --metrics-jsonl "$REPORT_DIR/cbddb_${ROUND_TAG}_distill_metrics.jsonl" \
  --out "$REPORT_DIR/cbddb_${ROUND_TAG}_distill_train.json" \
  >> "$LOG_DIR/cbddb_${ROUND_TAG}_train.log" 2>&1; then
  hb "TRAIN COMPLETE"
else
  hb "TRAIN FAILED"; exit 1
fi

ROUND_MANIFEST="cascadiav3/checkpoints/cbddb_${ROUND_TAG}_distill/best_locked_val.manifest.json"
test -s "$ROUND_MANIFEST"

report="$REPORT_DIR/cbddb_${ROUND_TAG}_screen_n256_d4.json"
if [ -s "$report" ]; then
  hb "EVAL screen reuse"
else
  hb "EVAL screen_n256_d4 starting (n256/d4 x 100)"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" \
    --manifest "$ROUND_MANIFEST" \
    --scoring-cards cbddb \
    --device cuda \
    --first-seed "$EVAL_FIRST_SEED" \
    --games 100 \
    --jobs "$EVAL_JOBS" \
    --batch-runner \
    --gumbel-n-simulations 256 \
    --gumbel-top-m 16 \
    --gumbel-depth-rounds 1 \
    --gumbel-determinizations 4 \
    --gumbel-market-decision-samples 8 \
    --gumbel-blend-weight 0.5 \
    --k-interior 16 \
    --control none \
    --model-timeout-ms 300000 \
    --source-revision "$SOURCE_REVISION" \
    --experiment-id "cbddb_${ROUND_TAG}_screen_n256_d4" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/cbddb_${ROUND_TAG}_screen_n256_d4_decisions.jsonl" \
    --games-out "$REPORT_DIR/cbddb_${ROUND_TAG}_screen_n256_d4_games.jsonl" \
    --summary-out "$REPORT_DIR/cbddb_${ROUND_TAG}_screen_n256_d4.md"
  hb "EVAL screen_n256_d4 DONE"
fi

MEAN=$("$PYTHON" -c "
import json
r = json.load(open('$report'))
print(r['strategies']['gumbel-search']['mean_seat_score'])
")
hb "SCREEN mean_seat_score=$MEAN"
hb "ROUND $ROUND_TAG COMPLETE"
