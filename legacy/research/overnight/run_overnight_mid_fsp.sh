#!/bin/bash
# Overnight training + Modal HH validation.
#
# Mid-features FSP reservoir: 10,862 features (all v3 minus per-cell adjacency).
# 5.6M params, ~21MB weights. Balanced field of unexploitable opponents + mce93.
# 5 iterations × 100K games × 15 epochs, LR 1e-4 → 3e-5 decay, ε=0.1.
#
# After training: Modal HH validation of iter3 and iter5 vs 3× mce93.
# Results written to overnight/MORNING_SUMMARY.md.

set -eu

TAG=mid_fsp
CLI=./target-mid/release/cascadia-cli
SUMMARY=overnight/MORNING_SUMMARY.md

echo "═══ Overnight Run: Mid-Features FSP ═══"
echo "  Start: $(date)"
echo "  Binary: $CLI"
echo "  Tag: $TAG"
echo "  Features: 10,862 (v3 minus per-cell adjacency)"
echo

# ── Phase 1: Training ──
TAG="$TAG" CLI="$CLI" ./overnight/train_fsp_reservoir.sh 5

echo
echo "═══ Training complete at $(date) ═══"
echo

# ── Phase 2: Modal HH validation ──
# Validate iter3 (mid-checkpoint) and iter5 (final)
for iter in 3 5; do
  weights="nnue_weights_${TAG}_iter${iter}.bin"
  log="overnight/hh_${TAG}_iter${iter}.log"

  if [ ! -f "$weights" ]; then
    echo "  SKIP iter${iter}: weights not found ($weights)"
    continue
  fi

  echo "  Validating iter${iter} on Modal (N=52, vs 3×mce93)..."
  python3 -m modal run overnight/head_to_head_modal.py \
    --strategies "mce_new,mce_anchor,mce_anchor,mce_anchor" \
    --strategy-weights "mce_new=${weights},mce_anchor=nnue_weights_mce93.bin" \
    --game-samples 13 \
    --weights nnue_weights_mce93.bin \
    > "$log" 2>&1

  echo "  iter${iter} HH done. Results in $log"
done

# Also run 4-way: mce93 vs best iter vs legacy_fsp_iter1 vs sym_pool_iter1
echo "  Running 4-way diagnostic (mce93, mid_fsp_iter5, legacy_fsp_iter1, sym_pool_iter1)..."
fourway_log="overnight/hh_4way_overnight.log"
if [ -f "nnue_weights_${TAG}_iter5.bin" ]; then
  python3 -m modal run overnight/head_to_head_modal.py \
    --strategies "mce_mce93,mce_mid5,mce_legacy1,mce_pool1" \
    --strategy-weights "mce_mce93=nnue_weights_mce93.bin,mce_mid5=nnue_weights_${TAG}_iter5.bin,mce_legacy1=nnue_weights_legacy_fsp_iter1.bin,mce_pool1=nnue_weights_sym_pool_iter1.bin" \
    --game-samples 13 \
    --weights nnue_weights_mce93.bin \
    > "$fourway_log" 2>&1
  echo "  4-way done. Results in $fourway_log"
fi

# ── Phase 3: Write morning summary ──
echo
echo "═══ Writing summary to $SUMMARY ═══"

cat > "$SUMMARY" << 'HEADER'
# Overnight Training Summary

## Experiment: Mid-Features FSP Reservoir

Architecture: 10,862 features → 512 → 64 → 1 (~21 MB, 5.6M params)
— all v3 features MINUS per-cell adjacency (34K features / 17.6M params removed).

Opponents: FSP reservoir = {random, scarcity, preference, mce93} + past iters.
Recipe: 5 iters × 100K games × 15 epochs, LR 1e-4 → 3e-5, ε=0.1.

## Per-Iter Training RMSE

HEADER

for i in 1 2 3 4 5; do
  log="overnight/train_${TAG}_iter${i}.log"
  if [ -f "$log" ]; then
    rmse=$(grep "Final RMSE" "$log" | awk '{print $NF}')
    size=$(ls -lh "nnue_weights_${TAG}_iter${i}.bin" 2>/dev/null | awk '{print $5}')
    echo "| iter${i} | RMSE ${rmse:-N/A} | ${size:-N/A} |" >> "$SUMMARY"
  fi
done

echo "" >> "$SUMMARY"

# Append HH results
for iter in 3 5; do
  log="overnight/hh_${TAG}_iter${iter}.log"
  if [ -f "$log" ]; then
    echo "## HH Validation: iter${iter} vs 3×mce93 (N=52)" >> "$SUMMARY"
    echo '```' >> "$SUMMARY"
    grep -A20 "TOURNAMENT RESULTS" "$log" | head -25 >> "$SUMMARY"
    echo '```' >> "$SUMMARY"
    echo "" >> "$SUMMARY"
  fi
done

if [ -f "$fourway_log" ]; then
  echo "## 4-Way Diagnostic: mce93 vs mid_fsp_iter5 vs legacy_fsp_iter1 vs sym_pool_iter1" >> "$SUMMARY"
  echo '```' >> "$SUMMARY"
  grep -A20 "TOURNAMENT RESULTS" "$fourway_log" | head -25 >> "$SUMMARY"
  echo '```' >> "$SUMMARY"
fi

echo "" >> "$SUMMARY"
echo "Completed: $(date)" >> "$SUMMARY"

echo
echo "═══ Overnight run complete at $(date) ═══"
echo "  Summary: $SUMMARY"
