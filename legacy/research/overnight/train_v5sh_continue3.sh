#!/bin/bash
# v5sh continuation training — iters 31-40 with continued LR decay.
#
# Hyperparameter schedule (linear interpolation across 10 iters):
#   LR:      3e-6 → 1e-6   (continued reduction beyond iter 30's 3e-6)
#   Epsilon: 0.03 → 0.02   (continued mild annealing)

set -eu
CLI=${CLI:-target-mid-v5/release/cascadia-cli}
NUM_ITERS=10
GAMES_PER_ITER=50000
START_ITER=31
EPOCHS=15
LR_START=3e-6
LR_END=1e-6
EPSILON_START=0.03
EPSILON_END=0.02
TAG=v5sh
NUM_FEATURES=17115
HIDDEN1=512
HIDDEN2=64
BATCH_SIZE=4096
INIT_WEIGHTS="nnue_weights_v5sh_iter$((START_ITER - 1)).bin"
BASE_POOL="random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin,nnue_weights_v4opp_modal_iter3.bin"

OUT_DIR=${OUT_DIR:-overnight/v5sh}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then echo "ERROR: binary not found at $CLI"; exit 1; fi
if [ ! -f "$INIT_WEIGHTS" ]; then echo "ERROR: init weights not found: $INIT_WEIGHTS"; exit 1; fi

interp() {
  python3 -c "print('{:.6e}'.format($1 + ($2 - $1) * $3 / ($4 - 1)))"
}

echo "════════════════════════════════════════════"
echo "  v5sh continuation training (iters $START_ITER..$((START_ITER + NUM_ITERS - 1)))"
echo "════════════════════════════════════════════"
echo "  Init from: $INIT_WEIGHTS (RMSE 3.99)"
echo "  LR schedule:      $LR_START → $LR_END (linear)"
echo "  Epsilon schedule: $EPSILON_START → $EPSILON_END (linear)"
echo

prev_weights="$INIT_WEIGHTS"
prior_iters=""
for j in $(seq 1 $((START_ITER - 1))); do
  prior_iters="${prior_iters},nnue_weights_${TAG}_iter${j}.bin"
done

for step in $(seq 1 $NUM_ITERS); do
  iter=$((START_ITER + step - 1))
  idx=$((step - 1))
  LR=$(interp "$LR_START" "$LR_END" "$idx" "$NUM_ITERS")
  EPSILON=$(interp "$EPSILON_START" "$EPSILON_END" "$idx" "$NUM_ITERS")
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  combined="$OUT_DIR/selfplay_${TAG}_iter${iter}.bin"
  pool="${BASE_POOL}${prior_iters}"

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  LR=$LR, Epsilon=$EPSILON"

  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED=$((4217 + iter * 7919)) \
    "$CLI" "$GAMES_PER_ITER" --selfplay-pool \
      --init-weights "$prev_weights" --out "$combined" --epsilon "$EPSILON" \
      > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1
  sp_samples=$(grep "SAMPLES=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  sp_elapsed=$(grep "ELAPSED_SEC=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Selfplay: ${sp_samples} samples in $(printf '%.0f' "${sp_elapsed:-0}")s"

  python3 train_pytorch.py value \
    --samples "$combined" --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH_SIZE" \
    --hidden1 "$HIDDEN1" --hidden2 "$HIDDEN2" --num-features "$NUM_FEATURES" \
    --optimizer sgd --no-augment \
    --init-weights "$prev_weights" --out "$new_weights" \
    > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED — halting"; tail -30 "$OUT_DIR/train_${TAG}_iter${iter}.log"; exit 1
  fi
  final_rmse=$(grep "RMSE=" "$OUT_DIR/train_${TAG}_iter${iter}.log" | tail -1 | grep -oE 'RMSE=[0-9.]+' | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=${final_rmse}"
  echo
  prev_weights="$new_weights"
  prior_iters="${prior_iters},${new_weights}"
done

echo "════════════════════════════════════════════"
echo "  Continuation complete — iters $START_ITER..$((START_ITER + NUM_ITERS - 1))"
echo "════════════════════════════════════════════"
