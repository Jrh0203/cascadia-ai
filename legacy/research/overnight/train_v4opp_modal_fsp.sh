#!/bin/bash
# v4-opp FSP training with Modal-parallelized data gen and local training.
#
# Each iter:
#   1. Fan out N workers on Modal, each producing a shard via --selfplay-pool
#   2. Concatenate shards into combined.bin (~50K games × 20 samples = 1M)
#   3. Local: cache-train on combined.bin, save iter weights
#   4. Upload new weights to Modal volume (via content-addressed cache, automatic)
#   5. Add new weights to opp pool for next iter
#
# Usage: ./overnight/train_v4opp_modal_fsp.sh [num_iters] [games_per_iter]

set -eu
CLI=${CLI:-./target-mid-v4/release/cascadia-cli}
NUM_ITERS=${1:-3}
GAMES_PER_ITER=${2:-50000}
NUM_WORKERS=${NUM_WORKERS:-50}
EPOCHS=${EPOCHS:-10}
LR=${LR:-3e-5}
EPSILON=${EPSILON:-0.1}
TAG=${TAG:-v4opp_modal}
INIT_WEIGHTS=${INIT_WEIGHTS:-nnue_weights_v4opp_fsp_iter3.bin}
BASE_POOL=${BASE_POOL:-"random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin,nnue_weights_v4opp_fsp_iter3.bin"}
PLAYER_MCE=${PLAYER_MCE:-0}  # 0 = NNUE-argmax; >0 = on-policy MCE(N) for player 0

OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then
  echo "ERROR: binary not found at $CLI. Build with:"
  echo "  cargo build --release --features mid-features,v4-opp --bin cascadia-cli --target-dir target-mid-v4"
  exit 1
fi

echo "════════════════════════════════════════════"
echo "  v4-opp Modal FSP training"
echo "════════════════════════════════════════════"
echo "  Tag: $TAG"
echo "  Iterations: $NUM_ITERS × $GAMES_PER_ITER games"
echo "  Workers: $NUM_WORKERS, Epochs: $EPOCHS, LR: $LR, ε: $EPSILON"
echo "  Init weights: $INIT_WEIGHTS"
echo "  Base pool: $BASE_POOL"
echo

prev_weights="$INIT_WEIGHTS"

for iter in $(seq 1 $NUM_ITERS); do
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  combined="$OUT_DIR/selfplay_${TAG}_iter${iter}.bin"

  # Build pool: base + all prior iters.
  # Guard against BSD `seq` which counts down when end < start (e.g. `seq 1 0`
  # yields "1 0" on macOS rather than empty like GNU seq).
  pool="$BASE_POOL"
  if [ "$iter" -gt 1 ]; then
    for j in $(seq 1 $((iter - 1))); do
      pool="${pool},nnue_weights_${TAG}_iter${j}.bin"
    done
  fi

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  Init from: $prev_weights"
  echo "  Pool: $pool"
  echo "  Combined cache: $combined"

  # STEP 1: Modal parallel selfplay
  echo "  [$(date +%H:%M:%S)] Modal selfplay dispatch..."
  python3 -m modal run overnight/selfplay_fsp_modal.py \
    --total-games "$GAMES_PER_ITER" \
    --num-workers "$NUM_WORKERS" \
    --weights "$prev_weights" \
    --opp-pool "$pool" \
    --out "$combined" \
    --epsilon "$EPSILON" \
    --seed-base "$((4217 + iter * 7919))" \
    --player-mce "$PLAYER_MCE" \
    > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1
  echo "  [$(date +%H:%M:%S)] Modal done. Log: $OUT_DIR/selfplay_${TAG}_iter${iter}.log"
  ls -lh "$combined"

  # STEP 2: Local cache-train
  echo "  [$(date +%H:%M:%S)] Local cache-train..."
  "$CLI" 0 --cache-train \
    --cache "$combined" \
    --init-weights "$prev_weights" \
    --weights "$new_weights" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED: no weights saved — halting"
    tail -30 "$OUT_DIR/train_${TAG}_iter${iter}.log"
    exit 1
  fi

  final_rmse=$(grep "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | awk '{print $NF}' | tail -1)
  size=$(stat -f%z "$new_weights" 2>/dev/null || stat -c%s "$new_weights" 2>/dev/null)
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=$final_rmse, $(du -h $new_weights | awk '{print $1}')"
  echo "  Saved: $new_weights"
  echo

  prev_weights="$new_weights"
done

echo "════════════════════════════════════════════"
echo "  Training complete — $NUM_ITERS iterations"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
