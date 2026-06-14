#!/bin/bash
# Phase 3: variations that build on the LEAF eval winner.
# Run AFTER leaf2 / leaf_200g / baseline_200g all complete.

set -e
WEIGHTS=nnue_weights_hybrid_iter4.bin
mkdir -p bench_results

run_bench() {
    local name=$1
    shift
    local outfile="bench_results/${name}.log"
    if [ -f "$outfile" ] && grep -q "Mean:" "$outfile" 2>/dev/null; then
        echo "[skip] $name"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] $name"
    "$@" > "$outfile" 2>&1
    local mean=$(grep -m1 "Mean:" "$outfile" | head -1 | awk '{print $2}')
    local time=$(grep -m1 "Results" "$outfile" | sed 's/.*in //; s/,.*//')
    echo "[$(date +%H:%M:%S)] Done $name → mean=$mean time=$time"
}

# 1. LEAF + deeper rollouts (depth=8)
MCE_LEAF_EXPECTIMAX=1 MCE_DEPTH=8 run_bench "leaf_d8_50g" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 2. LEAF + bigger budget (1500 rollouts)
MCE_LEAF_EXPECTIMAX=1 run_bench "leaf_n1500_50g" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 1500

# 3. LEAF + more candidates (MCE_CANDIDATES=20)
MCE_LEAF_EXPECTIMAX=1 MCE_CANDIDATES=20 run_bench "leaf_c20_50g" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 4. LEAF + smaller depth (4) — test if shorter rollouts help
MCE_LEAF_EXPECTIMAX=1 MCE_DEPTH=4 run_bench "leaf_d4_50g" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 5. LEAF2 v2 200g confirmation (only if leaf2_v2 50g shows promise)
# Conditional — runs only if leaf2_v2 mean is >= 96.0
if [ -f bench_results/leaf2_v2.log ]; then
    leaf2_mean=$(grep -m1 "Mean:" bench_results/leaf2_v2.log 2>/dev/null | awk '{print $2}')
    if [ -n "$leaf2_mean" ]; then
        is_better=$(awk "BEGIN { print ($leaf2_mean >= 96.0) ? 1 : 0 }")
        if [ "$is_better" = "1" ]; then
            MCE_LEAF_EXPECTIMAX2=1 run_bench "leaf2_200g" \
                ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750
        fi
    fi
fi

echo "[$(date +%H:%M:%S)] Phase 3 complete"
