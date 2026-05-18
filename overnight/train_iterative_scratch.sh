#!/bin/bash
# From-scratch iterative training with frozen mce93 opponents.
# Same anchor mechanism as sym_v2 but starts from random weights.
# Seeded per iter for diversity.
set -u
NUM_ITERS=${1:-5}
ANCHOR=${ANCHOR:-nnue_weights_mce93.bin}
TAG=${TAG:-sym_v3}
GAMES_PER_ITER=${GAMES_PER_ITER:-10000}
EPOCHS=${EPOCHS:-10}
LR=${LR:-0.0001}
VAL_SAMPLES=${VAL_SAMPLES:-3}

OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

# Remove any stale sym_v3 weights so iter 1 starts fresh
rm -f "nnue_weights_${TAG}_iter"*.bin

CURRENT_BEST=""
CURRENT_ANCHOR="$ANCHOR"

echo "═══ From-Scratch Iterative Training ═══"
echo "  Tag: $TAG"
echo "  Anchor (opponent, fixed): $CURRENT_ANCHOR"
echo "  Games/iter: $GAMES_PER_ITER, Epochs: $EPOCHS, LR: $LR"
echo "  Iterations: $NUM_ITERS"
echo

for iter in $(seq 1 $NUM_ITERS); do
  prev=$((iter - 1))
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  prev_weights="nnue_weights_${TAG}_iter${prev}.bin"

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  Opponents: $CURRENT_ANCHOR"
  echo "  Output:    $new_weights"

  seed=$((42 + iter * 7919))
  if [ $iter -eq 1 ]; then
    # First iter: fresh random weights (no --init-weights)
    echo "  Init: FRESH RANDOM"
    CASCADIA_TRAIN_OPP_WEIGHTS="$CURRENT_ANCHOR" \
    CASCADIA_TRAIN_SEED="$seed" \
      ./target/release/cascadia-cli "$GAMES_PER_ITER" --nnue-train \
        --lr "$LR" --epochs "$EPOCHS" \
        --weights "$new_weights" \
        > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1
  else
    echo "  Init from: $prev_weights"
    CASCADIA_TRAIN_OPP_WEIGHTS="$CURRENT_ANCHOR" \
    CASCADIA_TRAIN_SEED="$seed" \
      ./target/release/cascadia-cli "$GAMES_PER_ITER" --nnue-train \
        --lr "$LR" --epochs "$EPOCHS" \
        --init-weights "$prev_weights" \
        --weights "$new_weights" \
        > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1
  fi

  final_rmse=$(grep "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | awk '{print $NF}')
  echo "  Trained RMSE: $final_rmse"

  # Validate vs the ANCHOR (mce93), not prior iter. This is a fixed bar to climb.
  echo "  Validating vs anchor ($CURRENT_ANCHOR)..."
  val_log="$OUT_DIR/val_${TAG}_iter${iter}.log"
  python3 -u /Users/johnherrick/cascadia/overnight/head_to_head.py \
    --strategies "mce_new,mce_anchor,mce_anchor2,mce_anchor3" \
    --strategy-weights "mce_new=$new_weights,mce_anchor=$CURRENT_ANCHOR,mce_anchor2=$CURRENT_ANCHOR,mce_anchor3=$CURRENT_ANCHOR" \
    --game-samples "$VAL_SAMPLES" \
    > "$val_log" 2>&1

  new_wr=$(grep "^mce_new " "$val_log" | head -1 | awk '{print $3}' | tr -d '%')
  echo "  mce_new win rate vs anchor: ${new_wr}%"

  CURRENT_BEST="$new_weights"

  if [ -z "$new_wr" ]; then
    echo "  ERR: no win rate parsed"
  elif awk "BEGIN {exit !($new_wr >= 55)}"; then
    echo "  BEATS ANCHOR (${new_wr}%)"
  else
    echo "  BELOW ANCHOR (${new_wr}%, keep iterating)"
  fi

  echo
done

echo "═══ Training complete ═══"
echo "  Final: $CURRENT_BEST"
