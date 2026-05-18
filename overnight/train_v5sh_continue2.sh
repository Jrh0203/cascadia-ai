#!/bin/bash
# v5sh continuation training — iters 21-30 with continued LR decay.
#
# Hyperparameter schedule (linear interpolation across 10 iters):
#   LR:      1e-5 → 3e-6   (further reduction beyond iter 20's 1e-5)
#   Epsilon: 0.05 → 0.03   (mild further annealing)
#
# Rationale:
#   - Iters 11-20 already found big gains at LR 5e-5 → 1e-5 (RMSE 4.97 → 4.32)
#   - Last 3 iters' deltas were shrinking (0.068, 0.047) — entering fine-tune territory
#   - Going below 1e-5 may extract additional convergence
#   - Continued epsilon decay reduces exploration as model is close to converged

set -eu
CLI=${CLI:-target-mid-v5/release/cascadia-cli}
NUM_ITERS=10
GAMES_PER_ITER=50000
START_ITER=21
EPOCHS=15
LR_START=1e-5
LR_END=3e-6
EPSILON_START=0.05
EPSILON_END=0.03
TAG=v5sh
NUM_FEATURES=17115
HIDDEN1=512
HIDDEN2=64
BATCH_SIZE=4096
INIT_WEIGHTS="nnue_weights_v5sh_iter$((START_ITER - 1)).bin"
BASE_POOL="random,scarcity,preference,nnue_weights_mce93.bin,nnue_weights_mid_fsp_iter10.bin,nnue_weights_v4opp_modal_iter3.bin"

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

interp() {
  local start=$1 end=$2 idx=$3 total=$4
  python3 -c "print('{:.6e}'.format($start + ($end - $start) * $idx / ($total - 1)))"
}

echo "════════════════════════════════════════════"
echo "  v5sh continuation training (iters 21-30)"
echo "════════════════════════════════════════════"
echo "  Iterations: $NUM_ITERS × $GAMES_PER_ITER games (LOCAL)"
echo "  Init from: $INIT_WEIGHTS (RMSE 4.32)"
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
  echo "  Init from: $prev_weights"

  echo "  [$(date +%H:%M:%S)] Local selfplay ($GAMES_PER_ITER games, ε=$EPSILON)..."
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
  echo "  [$(date +%H:%M:%S)] Selfplay: ${sp_samples} samples (${sp_size}) in $(printf '%.0f' "${sp_elapsed:-0}")s"

  echo "  [$(date +%H:%M:%S)] Train (epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE)..."
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
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=${final_rmse}, size=${size}"
  echo

  prev_weights="$new_weights"
  prior_iters="${prior_iters},${new_weights}"
done

echo "════════════════════════════════════════════"
echo "  Continuation complete — iters $START_ITER..$((START_ITER + NUM_ITERS - 1))"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
