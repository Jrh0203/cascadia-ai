#!/bin/bash
# cards-alt-v3 NNUE training — FROM SCRATCH with fully-alt-aware pipeline.
#
# Hypothesis: warm-starting from cards_alt_v2_iter1 inherits whatever bias was
# learned with the Card-A-poisoned candidate generation during v2 training.
# Training from scratch with the FIXED pipeline (alt-aware candidates,
# alt-aware potential, gated-off inverted features, alt-aware-greedy
# opponents) means every training sample reflects the actual scoring rules
# from iter 1 onwards. Cleaner signal, but needs more iters.
#
# Architecture: 21,710 features → 512 → 64 → 1
# Init: from scratch (NNUENetwork::new(), random init)
#
# Pool: heuristics + iter15 + cards_alt_v2_iter1 from day 1. The two prior
# NNUE checkpoints play with their own learned policies (different from
# random) so the new network always faces a non-trivial opponent. Each new
# iter's checkpoint is added to the pool.
#
# Schedule (from scratch needs more iters to learn):
#   Phase 1 (iters 1-5):   25K × 5ep × LR 1e-4   (bootstrap)
#   Phase 2 (iters 6-12):  35K × 10ep × LR 5e-5 → 1e-5  (refine)
#   Phase 3 (iters 13-15): 35K × 10ep × LR 3e-6 → 1e-6  (polish)
#
# Total: ~525K games. Estimated wall ~8-12 hr (alt-aware adds per-iter cost).
#
# Usage: ./overnight/train_cards_alt_v3.sh

set -euo pipefail
CLI=${CLI:-target-mid-altv2/release/cascadia-cli}
TAG=${TAG:-cards_alt_v3}
NUM_FEATURES=21710
EPSILON=${EPSILON:-0.1}
SCORING_CARDS=${SCORING_CARDS:-C,B,D,D,B}
SEED_BASE=${SEED_BASE:-42}

# Pool: heuristics + 2 prior NNUE checkpoints from day 1.
BASE_POOL="random,scarcity,preference,nnue_weights_cards_alt_iter15.bin,nnue_weights_cards_alt_v2_iter1.bin"

OUT_DIR=${OUT_DIR:-overnight/cards_alt_v3}
mkdir -p "$OUT_DIR"

if [ ! -x "$CLI" ]; then
  echo "ERROR: binary missing at $CLI"; exit 1
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

  CASCADIA_SCORING_CARDS="$SCORING_CARDS" \
  CASCADIA_GREEDY_POTENTIAL=1 \
  CASCADIA_TRAIN_OPP_POOL="$pool" \
  CASCADIA_TRAIN_SEED=$((SEED_BASE + iter * 7919)) \
    "$CLI" "$games" --selfplay-pool \
      $init_arg --out "$combined" --epsilon "$EPSILON" \
      > "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" 2>&1

  local sp_samples=$(grep "SAMPLES=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  local sp_elapsed=$(grep "ELAPSED_SEC=" "$OUT_DIR/selfplay_${TAG}_iter${iter}.log" | head -1 | cut -d= -f2)
  echo "  [$(date +%H:%M:%S)] Selfplay: ${sp_samples} samples in $(printf '%.0f' "${sp_elapsed:-0}")s"

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
  local final_rmse=$(grep -E "Final RMSE" "$OUT_DIR/train_${TAG}_iter${iter}.log" | tail -1 | grep -oE 'RMSE[=: ]*[0-9.]+' | head -1 | grep -oE '[0-9.]+')
  echo "  [$(date +%H:%M:%S)] Trained. RMSE=${final_rmse}"
  echo
}

echo "════════════════════════════════════════════"
echo "  cards-alt-v3 NNUE training (15 iters FROM SCRATCH)"
echo "════════════════════════════════════════════"
echo "  Architecture: $NUM_FEATURES → 512 → 64 → 1"
echo "  Init: from scratch (random weights)"
echo "  Pool from day 1: heuristics + iter15 + v2_iter1"
echo "  Pipeline fixes baked in:"
echo "    - wildlife_strategic_candidates dispatched per scoring card"
echo "    - board_potential dispatched per animal-card"
echo "    - PAT_BEAR_WASTE / PAT_HAWK_AT_RISK / PAT_MAX_DIV_FOX gated off"
echo "    - CASCADIA_GREEDY_POTENTIAL=1 in selfplay"
echo "  Phases:"
echo "    P1 (iters 1-5):   25K × 5ep × LR 1e-4 (bootstrap)"
echo "    P2 (iters 6-12):  35K × 10ep × LR 5e-5 → 1e-5 (refine)"
echo "    P3 (iters 13-15): 35K × 10ep × LR 3e-6 → 1e-6 (polish)"
echo

prev=""

# Phase 1: bootstrap
for iter in 1 2 3 4 5; do
  run_iter "$iter" 25000 5 1e-4 "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 2: refine — 7 steps from 5e-5 → 1e-5
for step in 0 1 2 3 4 5 6; do
  iter=$((6 + step))
  lr=$(interp 5e-5 1e-5 "$step" 7)
  run_iter "$iter" 35000 10 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

# Phase 3: polish — 3 steps from 3e-6 → 1e-6
for step in 0 1 2; do
  iter=$((13 + step))
  lr=$(interp 3e-6 1e-6 "$step" 3)
  run_iter "$iter" 35000 10 "$lr" "$prev"
  prev="nnue_weights_${TAG}_iter${iter}.bin"
done

echo "════════════════════════════════════════════"
echo "  cards-alt-v3 training complete"
echo "════════════════════════════════════════════"
ls -lh nnue_weights_${TAG}_iter*.bin 2>/dev/null
echo "training complete"
