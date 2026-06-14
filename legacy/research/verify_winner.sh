#!/bin/bash
# Verify the LEAF1 winner with a clean A/B comparison.
# Run this when other benches are done — a clean comparison with the same seed offset.

set -e
WEIGHTS=nnue_weights_hybrid_iter4.bin
N=200

echo "=== Verification: LEAF1 vs baseline, $N games ==="
echo "Weights: $WEIGHTS"
echo "Same seed offset, same opponent strategy."
echo

mkdir -p verify_results

echo "[$(date +%H:%M:%S)] Running baseline..."
./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750 \
    > verify_results/baseline.log 2>&1
baseline_mean=$(grep -m1 "Mean:" verify_results/baseline.log | awk '{print $2}')
echo "[$(date +%H:%M:%S)] baseline = $baseline_mean"

echo "[$(date +%H:%M:%S)] Running LEAF1..."
MCE_LEAF_EXPECTIMAX=1 ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750 \
    > verify_results/leaf1.log 2>&1
leaf1_mean=$(grep -m1 "Mean:" verify_results/leaf1.log | awk '{print $2}')
echo "[$(date +%H:%M:%S)] leaf1 = $leaf1_mean"

delta=$(awk "BEGIN { printf \"%.2f\", $leaf1_mean - $baseline_mean }")
echo
echo "=== Result ==="
echo "Baseline:    $baseline_mean"
echo "LEAF1:       $leaf1_mean"
echo "Delta:       $delta"
echo
echo "200-game stderr ≈ ±0.28. Δ > 0.6 should be statistically significant."
