#!/bin/bash
# Fictitious Self-Play reservoir training.
#
# Opponents per seat sampled independently each game from a growing reservoir:
#   iter 1: {random, scarcity, preference, mce93}
#   iter 2: {random, scarcity, preference, mce93, <iter1 weights>}
#   iter 3: {random, scarcity, preference, mce93, <iter1>, <iter2>}
#   iter N: draft opponents ∪ {mce93} ∪ {all past iters}
#
# Gives the learner: (a) unexploitable draft opponents for diversity,
# (b) bear-competition signal from mce93, (c) competition against progressively
# stronger past-selves. This is AlphaStar-style league training applied to
# the value-function setting.
#
# Recipe otherwise matches original mce93 training (100K × 15 ep × LR 1e-4 × ε=0.1).

set -u
CLI=${CLI:-./target/release/cascadia-cli}
NUM_ITERS=${1:-5}
START_ITER=${START_ITER:-1}
TAG=${TAG:-sym_fsp}
GAMES_PER_ITER=${GAMES_PER_ITER:-100000}
EPOCHS=${EPOCHS:-15}
LR=${LR:-1e-4}
EPSILON=${EPSILON:-0.1}
SEED_WEIGHTS=${SEED_WEIGHTS:-nnue_weights_mce93.bin}  # seeds bear-competition signal
BASE_POOL=${BASE_POOL:-"random,scarcity,preference"}  # always-unexploitable anchors

OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

echo "═══ Fictitious Self-Play Reservoir Training ═══"
echo "  Tag: $TAG"
echo "  Base pool (always included): $BASE_POOL, $SEED_WEIGHTS"
echo "  Games/iter: $GAMES_PER_ITER, Epochs: $EPOCHS, LR: $LR, ε: $EPSILON"
echo "  Iterations: $NUM_ITERS"
echo

for iter in $(seq $START_ITER $NUM_ITERS); do
  prev=$((iter - 1))
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  prev_weights="nnue_weights_${TAG}_iter${prev}.bin"

  # Build reservoir pool for this iter: draft opponents + mce93 + all past iters.
  pool="$BASE_POOL,$SEED_WEIGHTS"
  if [ $prev -ge 1 ]; then
    for j in $(seq 1 $prev); do
      pool="${pool},nnue_weights_${TAG}_iter${j}.bin"
    done
  fi

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  Pool: $pool"
  echo "  Output: $new_weights"

  seed=$((42 + iter * 7919))
  init_arg=""
  if [ $iter -gt 1 ]; then
    init_arg="--init-weights $prev_weights"
    echo "  Init from: $prev_weights"
  else
    echo "  Init: FRESH RANDOM"
  fi

  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED="$seed" \
  CASCADIA_TRAIN_LR_DECAY="${LR_END:-3e-5}" \
    "$CLI" "$GAMES_PER_ITER" --nnue-train \
      --lr "$LR" --epochs "$EPOCHS" --epsilon "$EPSILON" \
      $init_arg \
      --weights "$new_weights" \
      > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  final_rmse=$(grep "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | awk '{print $NF}')
  echo "  Trained RMSE: $final_rmse"

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED: no weights file saved — halting"
    exit 1
  fi

  size=$(stat -f%z "$new_weights" 2>/dev/null || stat -c%s "$new_weights" 2>/dev/null)
  echo "  Saved: $new_weights (${size} bytes)"
  echo
done

echo "═══ FSP Training complete ═══"
ls -lh nnue_weights_${TAG}_iter*.bin
