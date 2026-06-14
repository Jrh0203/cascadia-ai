#!/bin/bash
# MCE config sweep against iter20 weights.
# Tests whether different MCE configurations work better with the new value network.

set -e
WEIGHTS=nnue_weights_hybrid_iter20.bin
mkdir -p bench_results

run_bench() {
    local name=$1
    shift
    local outfile="bench_results/${name}.log"
    if [ -f "$outfile" ] && grep -q "Mean:" "$outfile" 2>/dev/null; then
        echo "[skip] $name"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] Starting $name"
    "$@" > "$outfile" 2>&1
    local mean=$(grep -m1 "Mean:" "$outfile" | head -1 | awk '{print $2}')
    local time=$(grep -m1 "Results" "$outfile" | sed 's/.*in //; s/,.*//')
    echo "[$(date +%H:%M:%S)] Done $name → mean=$mean time=$time"
}

# Anchor: iter20 default at 200g (already done — leaf1_iter20_200g exists)
# Run sweep variants at 200g for statistical comparison

# 1. iter20 + depth=4
MCE_DEPTH=4 run_bench "iter20_d4_200g" \
    ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750

# 2. iter20 + 1500 rollouts
run_bench "iter20_n1500_200g" \
    ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 1500

# 3. iter20 + CANDIDATES=20
MCE_CANDIDATES=20 run_bench "iter20_c20_200g" \
    ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750

# 4. iter20 + depth=8 (control: did depth=8 hurt iter4? — test on iter20)
MCE_DEPTH=8 run_bench "iter20_d8_200g" \
    ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750

echo "[$(date +%H:%M:%S)] Sweep complete"
