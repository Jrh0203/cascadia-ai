#!/bin/bash
# v5-feat FSP training, from scratch, using Modal-parallelized data gen and
# local PyTorch training (split-11-value-head architecture).
#
# Each iter:
#   1. Modal selfplay (--selfplay-pool, v5-feat binary, MCV4 shards)
#   2. Local PyTorch training (train_pytorch.py value --split11-value-head)
#   3. Save iter weights, append to opp pool for next iter
#
# Usage: ./overnight/train_v5_fsp.sh [num_iters] [games_per_iter]

set -eu
NUM_ITERS=${1:-5}
GAMES_PER_ITER=${2:-30000}
NUM_WORKERS=${NUM_WORKERS:-100}
EPOCHS=${EPOCHS:-15}
LR=${LR:-1e-4}                # from-scratch is OK at 1e-4 (fine-tune diverges; see pipeline_params.md §12)
EPSILON=${EPSILON:-0.1}
TAG=${TAG:-v5sh}
NUM_FEATURES=17115            # mid + v4-opp + v5-feat
HIDDEN1=${HIDDEN1:-512}
HIDDEN2=${HIDDEN2:-64}
BATCH_SIZE=${BATCH_SIZE:-4096}

# Base FSP pool: stable opponents + the v4-opp champion (v4 weights load fine
# into v5 binary by zero-padding the 5,884 new v5-feat columns).
BASE_POOL=${BASE_POOL:-"random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin,nnue_weights_v4opp_modal_iter3.bin"}

OUT_DIR=${OUT_DIR:-overnight/v5feat}
mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════"
echo "  v5sh (single-head) FSP training — P3 ablation"
echo "════════════════════════════════════════════"
echo "  Tag: $TAG"
echo "  Iterations: $NUM_ITERS × $GAMES_PER_ITER games"
echo "  Workers: $NUM_WORKERS, Epochs: $EPOCHS, LR: $LR, ε: $EPSILON"
echo "  Architecture: $NUM_FEATURES → $HIDDEN1 → $HIDDEN2 → 1 head (single-head ablation)"
echo "  Base pool: $BASE_POOL"
echo

prev_weights=""    # empty = from-scratch on iter 1

for iter in $(seq 1 $NUM_ITERS); do
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  combined="$OUT_DIR/selfplay_${TAG}_iter${iter}.bin"

  # Build pool: base + all prior v5 iters
  pool="$BASE_POOL"
  if [ "$iter" -gt 1 ]; then
    for j in $(seq 1 $((iter - 1))); do
      pool="${pool},nnue_weights_${TAG}_iter${j}.bin"
    done
  fi

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  Init from: ${prev_weights:-'(from scratch)'}"
  echo "  Pool: $pool"
  echo "  Combined cache: $combined"

  # STEP 1: Modal parallel selfplay
  echo "  [$(date +%H:%M:%S)] Modal selfplay dispatch..."
  WEIGHTS_ARG=""
  if [ -n "$prev_weights" ]; then
    WEIGHTS_ARG="--weights $prev_weights"
  fi
  python3 -m modal run overnight/selfplay_fsp_v5_modal.py \
    --total-games "$GAMES_PER_ITER" \
    --num-workers "$NUM_WORKERS" \
    $WEIGHTS_ARG \
    --opp-pool "$pool" \
    --out "$combined" \
    --epsilon "$EPSILON" \
    --seed-base "$((4217 + iter * 7919))" \
    > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1
  echo "  [$(date +%H:%M:%S)] Modal done. Log: $OUT_DIR/selfplay_${TAG}_iter${iter}.log"
  ls -lh "$combined"

  # STEP 2: Local PyTorch training (single-head, MCV4 target=bonus-included)
  echo "  [$(date +%H:%M:%S)] Local PyTorch train (epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE)..."
  INIT_ARG=""
  if [ -n "$prev_weights" ]; then
    INIT_ARG="--init-weights $prev_weights"
  fi
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
    $INIT_ARG \
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
  echo "  Saved: $new_weights"
  echo

  prev_weights="$new_weights"
done

echo "════════════════════════════════════════════"
echo "  Training complete — $NUM_ITERS iterations"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
