#!/bin/bash
# Iterative self-play training with frozen-opponent anchoring + head-to-head gating.
#
# Each iter:
#   1. Train from iterN-1 weights, opponents = FROZEN anchor (default: mce93)
#   2. Validate iterN vs iterN-1 head-to-head (N=12 games)
#   3. Promote iterN if wins ≥55%, else keep iterN-1
#
# Anchor refresh: if iterN beats iterN-1 ≥70% (dominance), the anchor updates
# to the new weights for iter N+1 onward. This lets the opponent level rise
# as the trained model improves.
#
# Usage:
#   ANCHOR=nnue_weights_mce93.bin TAG=sym_v2 ./train_iterative_selfplay.sh 5

set -u
NUM_ITERS=${1:-5}
ANCHOR=${ANCHOR:-nnue_weights_mce93.bin}
TAG=${TAG:-sym_v2}
GAMES_PER_ITER=${GAMES_PER_ITER:-10000}
EPOCHS=${EPOCHS:-10}
LR=${LR:-0.0001}
VAL_SAMPLES=${VAL_SAMPLES:-3}  # game-samples × 4 rotations = 12 games

OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

# Seed weights (iter 0 = starting point)
cp "$ANCHOR" "nnue_weights_${TAG}_iter0.bin"
CURRENT_BEST="nnue_weights_${TAG}_iter0.bin"
CURRENT_ANCHOR="$ANCHOR"

echo "═══ Iterative Self-Play Training ═══"
echo "  Tag: $TAG"
echo "  Anchor (opponent): $CURRENT_ANCHOR"
echo "  Games/iter: $GAMES_PER_ITER, Epochs: $EPOCHS, LR: $LR"
echo "  Iterations: $NUM_ITERS"
echo "  Validation: $VAL_SAMPLES × 4 = $((VAL_SAMPLES * 4)) games/iter"
echo

for iter in $(seq 1 $NUM_ITERS); do
  prev=$((iter - 1))
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  prev_weights="nnue_weights_${TAG}_iter${prev}.bin"

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  Init from: $prev_weights"
  echo "  Opponents: $CURRENT_ANCHOR"
  echo "  Output:    $new_weights"

  # TRAIN: load from prev, save to new, opponents = anchor.
  # Seed varies per iter so rejected iters don't reproduce identical weights on retry.
  seed=$((42 + iter * 7919))
  CASCADIA_TRAIN_OPP_WEIGHTS="$CURRENT_ANCHOR" \
  CASCADIA_TRAIN_SEED="$seed" \
    ./target/release/cascadia-cli "$GAMES_PER_ITER" --nnue-train \
      --lr "$LR" --epochs "$EPOCHS" \
      --init-weights "$prev_weights" \
      --weights "$new_weights" \
      > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  # Extract final RMSE
  final_rmse=$(grep "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | awk '{print $NF}')
  echo "  Trained RMSE: $final_rmse"

  # VALIDATE: head-to-head iterN vs iterN-1 (4 seats, 3 copies of prev + 1 new for rotation)
  echo "  Validating vs $prev_weights..."
  val_log="$OUT_DIR/val_${TAG}_iter${iter}.log"
  python3 -u /Users/johnherrick/cascadia/overnight/head_to_head.py \
    --strategies "mce_new,mce_old,mce_old2,mce_old3" \
    --strategy-weights "mce_new=$new_weights,mce_old=$prev_weights,mce_old2=$prev_weights,mce_old3=$prev_weights" \
    --game-samples "$VAL_SAMPLES" \
    > "$val_log" 2>&1

  # Extract new-strategy win rate from the summary table (3rd column).
  # Must use head -1 because "mce_new" appears in multiple tables (wins, animals, ranks).
  new_wr=$(grep "^mce_new " "$val_log" | head -1 | awk '{print $3}' | tr -d '%')
  echo "  mce_new win rate: ${new_wr}%"

  # Decision: promote / keep / dominance
  if [ -z "$new_wr" ]; then
    echo "  ERR: no win rate parsed. Keeping prev weights."
    rm -f "$new_weights"
    cp "$prev_weights" "$new_weights"
  elif awk "BEGIN {exit !($new_wr >= 70)}"; then
    echo "  DOMINANCE ✓ (${new_wr}%) — promoting AND updating anchor"
    CURRENT_BEST="$new_weights"
    CURRENT_ANCHOR="$new_weights"
  elif awk "BEGIN {exit !($new_wr >= 55)}"; then
    echo "  PROMOTE ✓ (${new_wr}%) — keeping anchor, new becomes current best"
    CURRENT_BEST="$new_weights"
  else
    echo "  REJECT ✗ (${new_wr}%) — reverting to prev"
    rm -f "$new_weights"
    cp "$prev_weights" "$new_weights"
  fi

  echo "  Anchor: $CURRENT_ANCHOR  Best: $CURRENT_BEST"
  echo
done

echo "═══ Training complete ═══"
echo "  Final best: $CURRENT_BEST"
echo "  Anchor:     $CURRENT_ANCHOR"
