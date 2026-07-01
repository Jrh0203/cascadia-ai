#!/bin/bash
# v5sh continuation training — iters 11-20 with refined hyperparameters.
#
# Hyperparameter schedule (linear interpolation across 10 iters):
#   LR:      5e-5 → 1e-5
#   Epsilon: 0.10 → 0.05
#   Games per iter: 50K
#   Epochs: 15
#   Batch:  4096
#   Init: nnue_weights_v5sh_iter10.bin
#
# Rationale:
#   - LR decay addresses iter-10 RMSE bounce (1e-4 was too high near convergence)
#   - Epsilon decay refines play in the converged regime
#   - 10 iters give 2 buckets of 5 to confirm convergence
#
# Usage: ./overnight/train_v5sh_continue.sh

set -eu
CLI=${CLI:-target-mid-v5/release/cascadia-cli}
NUM_ITERS=10
GAMES_PER_ITER=50000
START_ITER=11
EPOCHS=15
LR_START=5e-5
LR_END=1e-5
EPSILON_START=0.10
EPSILON_END=0.05
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

# Per-iter linear interpolation helper. Uses python because bash lacks
# fractional arithmetic for scientific notation.
interp() {
  local start=$1 end=$2 idx=$3 total=$4
  python3 -c "print('{:.6e}'.format($start + ($end - $start) * $idx / ($total - 1)))"
}

echo "════════════════════════════════════════════"
echo "  v5sh continuation training (iters 11-20)"
echo "════════════════════════════════════════════"
echo "  Iterations: $NUM_ITERS × $GAMES_PER_ITER games (LOCAL)"
echo "  Init from: $INIT_WEIGHTS"
echo "  LR schedule:      $LR_START → $LR_END (linear)"
echo "  Epsilon schedule: $EPSILON_START → $EPSILON_END (linear)"
echo "  Architecture: $NUM_FEATURES → $HIDDEN1 → $HIDDEN2 → 1 head"
echo "  Base pool: $BASE_POOL"
echo "  + each prior v5sh iter (iter1..iter10 + new ones)"
echo

prev_weights="$INIT_WEIGHTS"
prior_iters=""
for j in $(seq 1 $((START_ITER - 1))); do
  prior_iters="${prior_iters},nnue_weights_${TAG}_iter${j}.bin"
done

for step in $(seq 1 $NUM_ITERS); do
  iter=$((START_ITER + step - 1))
  idx=$((step - 1))                                  # 0-indexed for interpolation
  LR=$(interp "$LR_START" "$LR_END" "$idx" "$NUM_ITERS")
  EPSILON=$(interp "$EPSILON_START" "$EPSILON_END" "$idx" "$NUM_ITERS")
  new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  combined="$OUT_DIR/selfplay_${TAG}_iter${iter}.bin"
  pool="${BASE_POOL}${prior_iters}"

  echo "[$(date +%H:%M:%S)] === Iteration $iter ==="
  echo "  LR=$LR, Epsilon=$EPSILON"
  echo "  Init from: $prev_weights"

  # STEP 1: Local --selfplay-pool
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

  # STEP 2: Local PyTorch training (single value head, MCV4 target)
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
