#!/bin/bash
# v6-peak training — local from-scratch, 20 iters, three-phase LR schedule.
#
# Phase 1 (iters 1-5):   bootstrap, 30K games × 5 epochs × LR=1e-4
# Phase 2 (iters 6-15):  mid,       50K games × 15 epochs × LR=5e-5 → 1e-5
# Phase 3 (iters 16-20): refine,    50K games × 15 epochs × LR=3e-6 → 1e-6
#
# FSP pool starts with heuristics + legacy NNUEs (mce93/mid_fsp/v4opp/v5sh_iter40),
# adds each prior v6 iter's checkpoint as training progresses.
#
# Total estimated wall: ~7-9 hours. Cost: $0 (all local).
#
# Usage: ./overnight/train_v6_local.sh

set -euo pipefail
CLI=${CLI:-target-mid-v6/release/cascadia-cli}
TAG=${TAG:-v6peak}
NUM_FEATURES=17608      # v6-peak total
HIDDEN1=${HIDDEN1:-512}
HIDDEN2=${HIDDEN2:-64}
BATCH_SIZE=${BATCH_SIZE:-4096}
EPSILON=${EPSILON:-0.1}

# FSP pool: heuristics only initially. Legacy v5/v4/v3-trained weights would
# load into v6 binary via byte-count math BUT their feature indices mean
# different things in v6's bounded layout — the loaded weights are garbage for
# v6 inference. So we exclude them; only v6-native checkpoints get added as
# training progresses.
BASE_POOL="random,scarcity,preference"

OUT_DIR=${OUT_DIR:-overnight/v6peak}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then echo "ERROR: binary missing $CLI"; exit 1; fi

interp() {
  python3 -c "print('{:.6e}'.format($1 + ($2 - $1) * $3 / ($4 - 1)))"
}

run_iter() {
  local iter=$1
  local games=$2
  local epochs=$3
  local lr=$4
  local prev_weights=$5

  local new_weights="nnue_weights_${TAG}_iter${iter}.bin"
  local combined="$OUT_DIR/selfplay_${TAG}_iter${iter}.bin"
  local pool="$BASE_POOL"
  for ((j=1; j<iter; j++)); do
    local pi="nnue_weights_${TAG}_iter${j}.bin"
    if [ -f "$pi" ]; then pool="${pool},${pi}"; fi
  done

  local init_arg=""
  if [ -n "$prev_weights" ] && [ -f "$prev_weights" ]; then
    init_arg="--init-weights $prev_weights"
  fi

  echo "[$(date +%H:%M:%S)] === Iteration $iter (games=$games epochs=$epochs LR=$lr) ==="
  echo "  Init: ${prev_weights:-'(from scratch)'}"
  echo "  Pool entries: $(echo $pool | tr ',' '\n' | wc -l | tr -d ' ')"

  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED=$((4217 + iter * 7919)) \
    "$CLI" "$games" --selfplay-pool \
      $init_arg --out "$combined" --epsilon "$EPSILON" \
      > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1

  local sp_samples=$(grep "SAMPLES=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  local sp_elapsed=$(grep "ELAPSED_SEC=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Selfplay: ${sp_samples} samples in $(printf '%.0f' "${sp_elapsed:-0}")s"

  local init_pt=""
  if [ -n "$prev_weights" ] && [ -f "$prev_weights" ]; then
    init_pt="--init-weights $prev_weights"
  fi
  python3 train_pytorch.py value \
    --samples "$combined" --epochs "$epochs" --lr "$lr" --batch-size "$BATCH_SIZE" \
    --hidden1 "$HIDDEN1" --hidden2 "$HIDDEN2" --num-features "$NUM_FEATURES" \
    --optimizer sgd --no-augment $init_pt --out "$new_weights" \
    > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED — train log:"; tail -30 "$OUT_DIR/train_${TAG}_iter${iter}.log"; exit 1
  fi
  local final_rmse=$(grep "RMSE=" "$OUT_DIR/train_${TAG}_iter${iter}.log" | tail -1 | grep -oE 'RMSE=[0-9.]+' | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=${final_rmse}"
  echo
}

echo "════════════════════════════════════════════"
echo "  v6-peak local training (20 iters from scratch)"
echo "════════════════════════════════════════════"
echo "  Architecture: $NUM_FEATURES → $HIDDEN1 → $HIDDEN2 → 1 (single head)"
echo "  Epsilon: $EPSILON"
echo "  Phases:"
echo "    P1 (iters 1-5):   30K games × 5 epochs × LR=1e-4 (bootstrap from scratch)"
echo "    P2 (iters 6-15):  50K games × 15 epochs × LR=5e-5 → 1e-5 (linear)"
echo "    P3 (iters 16-20): 50K games × 15 epochs × LR=3e-6 → 1e-6 (linear)"
echo "  Total games: ~900K"
echo

prev=""

# Phase 1: bootstrap (iters 1-5)
for iter in 1 2 3 4 5; do
  run_iter "$iter" 30000 5 1e-4 "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 2: mid (iters 6-15)
for step in 0 1 2 3 4 5 6 7 8 9; do
  iter=$((6 + step))
  lr=$(interp 5e-5 1e-5 "$step" 10)
  run_iter "$iter" 50000 15 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 3: refine (iters 16-20)
for step in 0 1 2 3 4; do
  iter=$((16 + step))
  lr=$(interp 3e-6 1e-6 "$step" 5)
  run_iter "$iter" 50000 15 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

echo "════════════════════════════════════════════"
echo "  v6-peak training complete"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
