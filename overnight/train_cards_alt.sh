#!/bin/bash
# cards-alt NNUE training — local from-scratch, alt scoring set
# (Bear C, Elk B, Salmon D, Hawk D, Fox B).
#
# Architecture: 11,423 features → 512 → 64 → 1 (single value head, mirrors v4-opp)
# NO existing NNUE has the alt-card features, so we train from scratch.
#
# 3-phase schedule (matches the v4opp recipe shape, scaled for from-scratch):
#   Phase 1 (iters 1-5):   25K games × 5 epochs × LR 1e-4 — bootstrap
#   Phase 2 (iters 6-12):  35K games × 10 epochs × LR 5e-5 → 1e-5 — refine
#   Phase 3 (iters 13-15): 35K games × 10 epochs × LR 3e-6 → 1e-6 — polish
#
# Total: ~475K games. Estimated wall ~7-9 hr on M4 mini.
#
# Opponent pool starts with just heuristics + greedy-with-card-aware-potential
# (no pre-trained alt-card NNUEs to seed with). Each prior iter's checkpoint is
# added to the pool as training progresses.
#
# Usage:
#   ./overnight/train_cards_alt.sh

set -euo pipefail
CLI=${CLI:-target-mid-alt/release/cascadia-cli}
TAG=${TAG:-cards_alt}
NUM_FEATURES=11423        # mid + v4-opp + cards-alt
HIDDEN1=${HIDDEN1:-512}
HIDDEN2=${HIDDEN2:-64}
EPSILON=${EPSILON:-0.1}
SCORING_CARDS=${SCORING_CARDS:-C,B,D,D,B}

# Heuristics-only base pool (no pre-trained alt-card NNUEs to seed with).
# Each prior iter's checkpoint is appended as training progresses.
BASE_POOL="random,scarcity,preference"

OUT_DIR=${OUT_DIR:-overnight/cards_alt}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then
  echo "ERROR: binary missing at $CLI"
  echo "  Build with: CARGO_TARGET_DIR=target-mid-alt cargo build --release -p cascadia-cli --features mid-features,v4-opp,cards-alt"
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

  local init_arg=""
  if [ -n "$prev_weights" ] && [ -f "$prev_weights" ]; then
    init_arg="--init-weights $prev_weights"
  fi

  echo "[$(date +%H:%M:%S)] === Iteration $iter (games=$games epochs=$epochs LR=$lr) ==="
  echo "  Init: ${prev_weights:-'(from scratch)'}"
  echo "  Pool entries: $(echo $pool | tr ',' '\n' | wc -l | tr -d ' ')"
  echo "  Cards: $SCORING_CARDS"

  CASCADIA_SCORING_CARDS="$SCORING_CARDS" \
  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED=$((4217 + iter * 7919)) \
    "$CLI" "$games" --selfplay-pool \
      $init_arg --out "$combined" --epsilon "$EPSILON" \
      > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1

  local sp_samples=$(grep "SAMPLES=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  local sp_elapsed=$(grep "ELAPSED_SEC=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Selfplay: ${sp_samples} samples in $(printf '%.0f' "${sp_elapsed:-0}")s"

  # Train via Rust --cache-train (faster than pytorch for our scale, no GPU needed).
  CASCADIA_SCORING_CARDS="$SCORING_CARDS" \
    "$CLI" 0 --cache-train \
      --cache "$combined" \
      $init_arg \
      --weights "$new_weights" \
      --epochs "$epochs" \
      --lr "$lr" \
      > "$OUT_DIR/train_${TAG}_iter${iter}.log" 2>&1

  if [ ! -f "$new_weights" ]; then
    echo "  FAILED — train log:"; tail -30 "$OUT_DIR/train_${TAG}_iter${iter}.log"; exit 1
  fi
  local final_rmse=$(grep -E "Final RMSE|RMSE=" "$OUT_DIR/train_${TAG}_iter${iter}.log" | tail -1 | grep -oE 'RMSE[=: ]*[0-9.]+' | head -1 | grep -oE '[0-9.]+')
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=${final_rmse}"
  echo
}

echo "════════════════════════════════════════════"
echo "  cards-alt NNUE training (15 iters from scratch)"
echo "════════════════════════════════════════════"
echo "  Architecture: $NUM_FEATURES → $HIDDEN1 → $HIDDEN2 → 1"
echo "  Scoring cards: $SCORING_CARDS"
echo "  Epsilon: $EPSILON"
echo "  Phases:"
echo "    P1 (iters 1-5):   25K × 5ep × LR 1e-4 (bootstrap)"
echo "    P2 (iters 6-12):  35K × 10ep × LR 5e-5 → 1e-5 (refine)"
echo "    P3 (iters 13-15): 35K × 10ep × LR 3e-6 → 1e-6 (polish)"
echo

prev=""

# Phase 1: bootstrap (iters 1-5)
for iter in 1 2 3 4 5; do
  run_iter "$iter" 25000 5 1e-4 "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 2: refine (iters 6-12) — 7 steps from 5e-5 → 1e-5
for step in 0 1 2 3 4 5 6; do
  iter=$((6 + step))
  lr=$(interp 5e-5 1e-5 "$step" 7)
  run_iter "$iter" 35000 10 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 3: polish (iters 13-15) — 3 steps from 3e-6 → 1e-6
for step in 0 1 2; do
  iter=$((13 + step))
  lr=$(interp 3e-6 1e-6 "$step" 3)
  run_iter "$iter" 35000 10 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

echo "════════════════════════════════════════════"
echo "  cards-alt training complete"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
echo "training complete"
