#!/bin/bash
# Bench all overnight-trained weight files when they're stable.
# Runs after v3eps and v4 finish.

set -e
mkdir -p bench_results

bench_one() {
    local weights=$1
    local label=$2
    local bench_log="bench_results/${label}_nnue_200g.log"
    if [ -f "$bench_log" ] && grep -q "Mean:" "$bench_log"; then
        echo "[skip] $label NNUE 200g already done"
    else
        echo "[$(date +%H:%M:%S)] Benching $label NNUE-only 200g..."
        nice -n 5 ./target/release/cascadia-cli 200 --nnue --weights "$weights" \
            > "$bench_log" 2>&1
        local mean=$(grep -m1 "Mean:" "$bench_log" | awk '{print $2}')
        echo "[$(date +%H:%M:%S)] $label NNUE 200g: $mean"
    fi
}

bench_mce() {
    local weights=$1
    local label=$2
    local bench_log="bench_results/${label}_mce_100g.log"
    if [ -f "$bench_log" ] && grep -q "Mean:" "$bench_log"; then
        echo "[skip] $label MCE 100g already done"
    else
        echo "[$(date +%H:%M:%S)] Benching $label MCE 100g..."
        nice -n 5 ./target/release/cascadia-cli 100 --mce --rollouts 750 --weights "$weights" \
            > "$bench_log" 2>&1
        local mean=$(grep -m1 "Mean:" "$bench_log" | awk '{print $2}')
        echo "[$(date +%H:%M:%S)] $label MCE 100g: $mean"
    fi
}

# Wait for trainings to finish
echo "[$(date +%H:%M:%S)] Waiting for v3eps + v4 trainings to finish..."
while pgrep -f "train_hybrid.py" > /dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] All training done. Starting benches."

# Find latest weights for each variant
v3eps_latest=$(ls -t nnue_weights_v3eps_iter*.bin 2>/dev/null | head -1)
v4_latest=$(ls -t nnue_weights_v4_iter*.bin 2>/dev/null | head -1)

echo "Latest v3eps: $v3eps_latest"
echo "Latest v4:    $v4_latest"
echo

# Bench all
if [ -n "$v3eps_latest" ]; then
    label="v3eps_$(basename $v3eps_latest .bin | sed 's/nnue_weights_v3eps_//')"
    bench_one "$v3eps_latest" "$label"
    bench_mce "$v3eps_latest" "$label"
fi

if [ -n "$v4_latest" ]; then
    label="v4_$(basename $v4_latest .bin | sed 's/nnue_weights_v4_//')"
    bench_one "$v4_latest" "$label"
    bench_mce "$v4_latest" "$label"
fi

# Compare against v3 iter20 baseline
bench_one nnue_weights_v3_iter20.bin "v3_iter20_baseline"
bench_one nnue_weights_hybrid_iter20.bin "v1_iter20_baseline"

echo
echo "[$(date +%H:%M:%S)] All benches complete!"
echo
echo "=== SUMMARY ==="
for f in bench_results/v3eps_*_nnue_200g.log bench_results/v4_*_nnue_200g.log \
         bench_results/v1_iter20_baseline_nnue_200g.log bench_results/v3_iter20_baseline_nnue_200g.log; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .log | sed 's/_nnue_200g//')
    mean=$(grep -m1 "Mean:" "$f" | awk '{print $2}')
    printf "%-30s NNUE 200g = %s\n" "$label" "$mean"
done
for f in bench_results/v3eps_*_mce_100g.log bench_results/v4_*_mce_100g.log; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .log | sed 's/_mce_100g//')
    mean=$(grep -m1 "Mean:" "$f" | awk '{print $2}')
    printf "%-30s MCE  100g = %s\n" "$label" "$mean"
done
