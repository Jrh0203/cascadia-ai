#!/bin/bash
# Bench every nnue_weights_hybrid_iterN.bin in NNUE-only mode (200 games each).
# Reveals the per-iter improvement trajectory.

set -e
mkdir -p iter_history

for iter in 1 4 8 12 16 20; do
    weights="nnue_weights_hybrid_iter${iter}.bin"
    outfile="iter_history/nnue_iter${iter}_200g.log"
    if [ ! -f "$weights" ]; then
        echo "[skip] $weights doesn't exist"
        continue
    fi
    if [ -f "$outfile" ] && grep -q "Mean:" "$outfile" 2>/dev/null; then
        mean=$(grep -m1 "Mean:" "$outfile" | awk '{print $2}')
        echo "[have] iter$iter mean=$mean"
        continue
    fi
    echo "[$(date +%H:%M:%S)] iter$iter benching..."
    ./target/release/cascadia-cli 200 --nnue --weights "$weights" > "$outfile" 2>&1
    mean=$(grep -m1 "Mean:" "$outfile" | awk '{print $2}')
    echo "[$(date +%H:%M:%S)] iter$iter mean=$mean"
done

echo
echo "=== History summary ==="
for iter in 1 4 8 12 16 20; do
    outfile="iter_history/nnue_iter${iter}_200g.log"
    if [ -f "$outfile" ]; then
        mean=$(grep -m1 "Mean:" "$outfile" | awk '{print $2}')
        echo "iter${iter}: $mean"
    fi
done
