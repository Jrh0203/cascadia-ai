#!/bin/bash
# Phase 4: items that queue 1 won't reach for hours due to slow NRPA.
# These can run in parallel with queue 1's NRPA grinding.

set -e
WEIGHTS=nnue_weights_hybrid_iter4.bin
N=50
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

# 1. MCE_LEAF_MARKET2 (2-step market-aware) — likely strongest variant
MCE_LEAF_MARKET2=1 run_bench "leaf_market2" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750

# 2. MCE_LEAF_MARKET 200g confirmation (after 50g shows promise)
# Run conditionally if leaf_market 50g mean >= 96.0
if [ -f bench_results/leaf_market.log ]; then
    leaf_market_mean=$(grep -m1 "Mean:" bench_results/leaf_market.log 2>/dev/null | awk '{print $2}')
    if [ -n "$leaf_market_mean" ]; then
        is_better=$(awk "BEGIN { print ($leaf_market_mean >= 96.0) ? 1 : 0 }")
        if [ "$is_better" = "1" ]; then
            MCE_LEAF_MARKET=1 run_bench "leaf_market_200g" \
                ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750
        fi
    fi
fi

# 3. Pure GUMBEL_TOPK (no leaf) — control
MCE_GUMBEL_TOPK=1 run_bench "gumbel_topk_only" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750

echo "[$(date +%H:%M:%S)] Phase 4 complete"
