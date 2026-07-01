#!/bin/bash
# v5sh continuation training — LOCAL ONLY (no Modal).
#
# Continues from a starting checkpoint (defaults to v5sh_iter5) and runs more
# iters with MORE games per iter. Both selfplay AND training run locally.
#
# Each iter:
#   1. Local --selfplay-pool (uses all available CPU threads)
#   2. Local PyTorch training (single value head, MCV4 target field)
#
# Usage: ./overnight/train_v5sh_local.sh [num_iters] [games_per_iter] [start_iter]
#   num_iters: how many additional iters to run (default 5)
#   games_per_iter: games per iter (default 50000 — more than the 30K Modal runs)
#   start_iter: iter number for the FIRST new iter (default 6, meaning iter6..iter10)

set -eu
CLI=${CLI:-target-mid-v5/release/cascadia-cli}
NUM_ITERS=${1:-5}
GAMES_PER_ITER=${2:-50000}
START_ITER=${3:-6}              # continue numbering from here
EPOCHS=${EPOCHS:-15}
LR=${LR:-1e-4}
EPSILON=${EPSILON:-0.1}
TAG=${TAG:-v5sh}
NUM_FEATURES=17115
HIDDEN1=${HIDDEN1:-512}
HIDDEN2=${HIDDEN2:-64}
BATCH_SIZE=${BATCH_SIZE:-4096}

INIT_WEIGHTS=${INIT_WEIGHTS:-nnue_weights_v5sh_iter$((START_ITER - 1)).bin}
BASE_POOL=${BASE_POOL:-"random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin,nnue_weights_v4opp_modal_iter3.bin"}

OUT_DIR=${OUT_DIR:-overnight/v5sh}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then
  echo "ERROR: binary not found at $CLI"
  exit 1
fi
if [ ! -f "$INIT_WEIGHTS" ]; then
  echo "ERROR: init weights not found: $INIT_WEIGHTS"
  exit 1
fi

echo "════════════════════════════════════════════"
echo "  v5sh LOCAL continuation training"
echo "════════════════════════════════════════════"
echo "  Tag: $TAG"
echo "  Iterations: $NUM_ITERS × $GAMES_PER_ITER games (LOCAL)"
echo "  Start iter: $START_ITER (continuing from $INIT_WEIGHTS)"
echo "  Epochs: $EPOCHS, LR: $LR, ε: $EPSILON"
echo "  Architecture: $NUM_FEATURES → $HIDDEN1 → $HIDDEN2 → 1 head"
echo "  Base pool: $BASE_POOL"
echo

prev_weights="$INIT_WEIGHTS"

# Build prior-iters list (iter1..iter(START_ITER-1)) so the FSP pool already
# contains all earlier checkpoints from the Modal run.
prior_iters=""
for j in $(seq 1 $((START_ITER - 1))); do
  prior_iters="${prior_iters},nnue_weights_${TAG}_iter${j}.bin"
done

for step in $(seq 1 $NUM_ITERS); do
  iter=$((START_ITER + step - 1))
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  combined="$OUT_DIR/selfplay_${TAG}_iter${iter}.bin"
  pool="${BASE_POOL}${prior_iters}"

  echo "[$(date +%H:%M:%S)] === Iteration $iter (local) ==="
  echo "  Init from: $prev_weights"
  echo "  Pool entries: $(echo "$pool" | tr ',' '\n' | wc -l | tr -d ' ')"
  echo "  Combined cache: $combined"

  # STEP 1: Local --selfplay-pool. Rust binary uses all available threads.
  echo "  [$(date +%H:%M:%S)] Local selfplay ($GAMES_PER_ITER games)..."
  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED=$((4217 + iter * 7919)) \
    "$CLI" "$GAMES_PER_ITER" --selfplay-pool \
      --init-weights "$prev_weights" \
      --out "$combined" \
      --epsilon "$EPSILON" \
      > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1
  sp_size=$(du -h "$combined" | awk '{print $1}')
  sp_samples=$(grep "SAMPLES=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  sp_elapsed=$(grep "ELAPSED_SEC=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Selfplay done. ${sp_samples} samples (${sp_size}) in $(printf '%.0f' "${sp_elapsed:-0}")s"

  # STEP 2: Local PyTorch training (single value head — MCV4 target field is bonus-included sum).
  echo "  [$(date +%H:%M:%S)] Local PyTorch train (epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE)..."
  python3 train_pytorch.py value \
    --samples "$combined" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --hidden1 "$HIDDEN1" \
    --hidden2 "$HIDDEN2" \
    --num-features "$NUM_FEATURES" \
    --optimizer sgd \
    --no-augment \
    --init-weights "$prev_weights" \
    --out "$new_weights" \
    > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED: no weights saved — halting"
    tail -30 "$OUT_DIR/train_${TAG}_iter${iter}.log"
    exit 1
  fi

  final_rmse=$(grep "RMSE=" "$OUT_DIR/train_${TAG}_iter${iter}.log" | tail -1 | grep -oE 'RMSE=[0-9.]+' | head -1 | cut -d= -f2)
  size=$(du -h "$new_weights" | awk '{print $1}')
  echo "  [$(date +%H:%M:%S)] Trained. Final RMSE=${final_rmse}, size=${size}"

  prev_weights="$new_weights"
  prior_iters="${prior_iters},${new_weights}"
done

echo "════════════════════════════════════════════"
echo "  Local training complete — $NUM_ITERS iterations (iter${START_ITER}..iter$((START_ITER + NUM_ITERS - 1)))"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
