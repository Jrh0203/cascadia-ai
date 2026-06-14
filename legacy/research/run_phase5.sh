#!/bin/bash
# Phase 5: leaf_market variants. Run after leaf_market 50g confirms.

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

# Phase 5 amends: leaf_market didn't win (95.5 at 50g), so focus on amplifying LEAF1 instead.

# 1. LEAF1 + smaller depth (4) — test if shorter rollouts amplify the leaf eval signal
MCE_LEAF_EXPECTIMAX=1 MCE_DEPTH=4 run_bench "leaf1_d4" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 2. LEAF1 + more candidates (20)
MCE_LEAF_EXPECTIMAX=1 MCE_CANDIDATES=20 run_bench "leaf1_c20" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 3. LEAF1 + 1500 rollouts (more samples)
MCE_LEAF_EXPECTIMAX=1 run_bench "leaf1_n1500" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 1500

# 4. LEAF1 + smaller candidate pool (10)
MCE_LEAF_EXPECTIMAX=1 MCE_CANDIDATES=10 run_bench "leaf1_c10" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

echo "[$(date +%H:%M:%S)] Phase 5 complete"
