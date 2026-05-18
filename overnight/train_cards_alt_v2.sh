#!/bin/bash
# cards-alt-v2 NNUE training — per-piece relational features.
#
# Hypothesis: replacing histogram-style alt-card features with per-piece
# position-aware features will give the network direct credit assignment per
# placed wildlife, restoring the NNUE-as-intended design pattern (Stockfish
# HalfKP). The original cards-alt training plateaued at RMSE 6.14 because
# aggregated histograms destroyed positional signal.
#
# Architecture: 21,710 features → 512 → 64 → 1
#   = mid (10,862) + v4-opp (369) + cards-alt (192) + cards-alt-v2 (10,287)
#
# WARM START from nnue_weights_cards_alt_iter15.bin (zero-padded for the new
# ~10K columns). 8 iters with mixed pool including iter15 from day 1.
#
# Schedule:
#   Phase 1 (iters 1-3): 20K games × 5 epochs × LR 5e-5 — bootstrap new columns
#   Phase 2 (iters 4-6): 25K games × 8 epochs × LR 3e-5 → 1e-5 — refine
#   Phase 3 (iters 7-8): 25K games × 8 epochs × LR 3e-6 → 1e-6 — polish
#
# Total: ~6-9 hr local on M4 mini.
#
# Usage: ./overnight/train_cards_alt_v2.sh

set -euo pipefail
CLI=${CLI:-target-mid-altv2/release/cascadia-cli}
TAG=${TAG:-cards_alt_v2}
NUM_FEATURES=21710
HIDDEN1=${HIDDEN1:-512}
HIDDEN2=${HIDDEN2:-64}
EPSILON=${EPSILON:-0.1}
SCORING_CARDS=${SCORING_CARDS:-C,B,D,D,B}
SEED_BASE=${SEED_BASE:-42}

# Pool: heuristics + the prior cards-alt champion (iter15) from day 1, plus
# all cards_alt_v2 iters as they accumulate. iter15 in the pool means we never
# self-play against pure-random — there's always a strong NNUE opponent.
BASE_POOL="random,scarcity,preference,nnue_weights_cards_alt_iter15.bin"

OUT_DIR=${OUT_DIR:-overnight/cards_alt_v2}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then
  echo "ERROR: binary missing at $CLI"
  echo "  Build with: CARGO_TARGET_DIR=target-mid-altv2 cargo build --release -p cascadia-cli --features mid-features,v4-opp,cards-alt-v2"
  exit 1
fi

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

  echo "[$(date +%H:%M:%S)] === Iteration $iter (games=$games epochs=$epochs LR=$lr) ==="
  echo "  Init: $prev_weights"
  echo "  Pool entries: $(echo $pool | tr ',' '\n' | wc -l | tr -d ' ')"

  CASCADIA_SCORING_CARDS="$SCORING_CARDS" \
  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED=$((SEED_BASE + iter * 7919)) \
    "$CLI" "$games" --selfplay-pool \
      --init-weights "$prev_weights" \
      --out "$combined" --epsilon "$EPSILON" \
      > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1

  local sp_samples=$(grep "SAMPLES=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  local sp_elapsed=$(grep "ELAPSED_SEC=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Selfplay: ${sp_samples} samples in $(printf '%.0f' "${sp_elapsed:-0}")s"

  CASCADIA_SCORING_CARDS="$SCORING_CARDS" \
    "$CLI" 0 --cache-train \
      --cache "$combined" \
      --init-weights "$prev_weights" \
      --weights "$new_weights" \
      --epochs "$epochs" \
      --lr "$lr" \
      > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED — train log:"; tail -30 "$OUT_DIR/train_${TAG}_iter${iter}.log"; exit 1
  fi
  local final_rmse=$(grep -E "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | tail -1 | grep -oE 'RMSE[=: ]*[0-9.]+' | head -1 | grep -oE '[0-9.]+')
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=${final_rmse}"
  echo
}

echo "════════════════════════════════════════════"
echo "  cards-alt-v2 NNUE training (8 iters, warm start)"
echo "════════════════════════════════════════════"
echo "  Architecture: $NUM_FEATURES → $HIDDEN1 → $HIDDEN2 → 1"
echo "  Init: nnue_weights_cards_alt_iter15.bin (zero-padded for new columns)"
echo "  Scoring cards: $SCORING_CARDS"
echo "  Phases:"
echo "    P1 (iters 1-3): 20K × 5ep × LR 5e-5 (bootstrap new columns)"
echo "    P2 (iters 4-6): 25K × 8ep × LR 3e-5 → 1e-5 (refine)"
echo "    P3 (iters 7-8): 25K × 8ep × LR 3e-6 → 1e-6 (polish)"
echo

prev="nnue_weights_cards_alt_iter15.bin"

# Phase 1
for iter in 1 2 3; do
  run_iter "$iter" 20000 5 5e-5 "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 2: 3 steps, 3e-5 → 1e-5
for step in 0 1 2; do
  iter=$((4 + step))
  lr=$(interp 3e-5 1e-5 "$step" 3)
  run_iter "$iter" 25000 8 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 3: 2 steps, 3e-6 → 1e-6
for step in 0 1; do
  iter=$((7 + step))
  lr=$(interp 3e-6 1e-6 "$step" 2)
  run_iter "$iter" 25000 8 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

echo "════════════════════════════════════════════"
echo "  cards-alt-v2 training complete"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
echo "training complete"
