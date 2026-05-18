#!/bin/bash
# FSP training for v4-opp NNUE.
#
# Uses `target-mid-v4/release/cascadia-cli` (built with --features mid-features,v4-opp).
# Initializes from mid_fsp_iter10 (zero-padded for the 369 new opp-detail columns).
# Opponent pool uses a mix of draft (random/scarcity/preference), mce93, mid_fsp_iter10,
# and progressively past iterations (AlphaStar-style league).
#
# Compact recipe (for session-level iteration):
#   - GAMES_PER_ITER: 20000 (vs standard 100K) — faster signal
#   - EPOCHS: 10 (vs standard 15) — avoid overfit on smaller data
#   - NUM_ITERS: 5 — enough for v4 columns to populate meaningfully
#   - Total wall: ~2.5 hours on M1 Max (10 cores)
#
# Usage:
#   ./overnight/train_v4opp_fsp.sh [num_iters]

set -u
CLI=${CLI:-./target-mid-v4/release/cascadia-cli}
NUM_ITERS=${1:-5}
START_ITER=${START_ITER:-1}
TAG=${TAG:-v4opp_fsp}
GAMES_PER_ITER=${GAMES_PER_ITER:-20000}
EPOCHS=${EPOCHS:-10}
# Fine-tuning LR: 1e-4 diverged on init-from-mid_fsp_iter10. 3e-5 is stable.
LR=${LR:-3e-5}
LR_END=${LR_END:-1e-5}
EPSILON=${EPSILON:-0.1}
SEED_WEIGHTS=${SEED_WEIGHTS:-nnue_weights_mid_fsp_iter10.bin}
INIT_WEIGHTS=${INIT_WEIGHTS:-nnue_weights_mid_fsp_iter10.bin}
BASE_POOL=${BASE_POOL:-"random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin"}

OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then
  echo "ERROR: binary not found at $CLI. Build with:"
  echo "  cargo build --release --features mid-features,v4-opp --bin cascadia-cli --target-dir target-mid-v4"
  exit 1
fi

echo "═══ v4-opp FSP training ═══"
echo "  Binary: $CLI"
echo "  Tag: $TAG"
echo "  Init weights: $INIT_WEIGHTS"
echo "  Base pool: $BASE_POOL"
echo "  Recipe: ${GAMES_PER_ITER}g × ${EPOCHS}ep × LR ${LR}→${LR_END} × ε=${EPSILON}"
echo "  Iterations: $START_ITER..$NUM_ITERS"
echo

for iter in $(seq $START_ITER $NUM_ITERS); do
  prev=$((iter - 1))
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  prev_weights="nnue_weights_${TAG}_iter${prev}.bin"

  pool="$BASE_POOL"
  if [ $prev -ge 1 ]; then
    for j in $(seq 1 $prev); do
      pool="${pool},nnue_weights_${TAG}_iter${j}.bin"
    done
  fi

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  Pool: $pool"
  echo "  Output: $new_weights"

  seed=$((4217 + iter * 7919))
  if [ $iter -eq 1 ]; then
    init_arg="--init-weights $INIT_WEIGHTS"
    echo "  Init from: $INIT_WEIGHTS (mid_fsp_iter10 with v4 cols padded to zero)"
  else
    init_arg="--init-weights $prev_weights"
    echo "  Init from: $prev_weights"
  fi

  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED="$seed" \
  CASCADIA_TRAIN_LR_DECAY="$LR_END" \
    "$CLI" "$GAMES_PER_ITER" --nnue-train \
      --lr "$LR" --epochs "$EPOCHS" --epsilon "$EPSILON" \
      $init_arg \
      --weights "$new_weights" \
      > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  final_rmse=$(grep "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | awk '{print $NF}')
  echo "  Final RMSE: $final_rmse"

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED: no weights file saved — halting"
    exit 1
  fi

  size=$(stat -f%z "$new_weights" 2>/dev/null || stat -c%s "$new_weights" 2>/dev/null)
  echo "  Saved: $new_weights (${size} bytes)"
  echo
done

echo "═══ v4-opp FSP complete ═══"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
