#!/bin/bash
# Second queue: variants and confirmation runs.
# Run AFTER queue 1 completes.

set -e
WEIGHTS=nnue_weights_hybrid_iter4.bin
mkdir -p bench_results

run_bench() {
    local name=$1
    shift
    local outfile="bench_results/${name}.log"
    if [ -f "$outfile" ] && grep -q "Mean:" "$outfile" 2>/dev/null; then
        echo "[skip] $name (already complete)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] Starting $name"
    "$@" > "$outfile" 2>&1
    local mean=$(grep -m1 "Mean:" "$outfile" | head -1 | awk '{print $2}')
    local time=$(grep -m1 "Results" "$outfile" | sed 's/.*in //; s/,.*//')
    echo "[$(date +%H:%M:%S)] Done $name → mean=$mean time=$time"
}

# === Confirmation runs of the best Tier 2 techniques (200 games each) ===
# 200 games gives ~0.5 pt noise band — small enough to detect a 0.6 pt gain.

# 1. Tier 2 #4 confirmation: LEAF_EXPECTIMAX with 200 games
MCE_LEAF_EXPECTIMAX=1 run_bench "leaf_only_200g" \
    ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750

# 2. Baseline confirmation: 200 games
run_bench "baseline_200g" \
    ./target/release/cascadia-cli 200 --mce --weights $WEIGHTS --rollouts 750

# === New variations ===

# 3. MCE_CANDIDATES=20 with LEAF (more diversity at root)
MCE_LEAF_EXPECTIMAX=1 MCE_CANDIDATES=20 run_bench "leaf_c20" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 4. MCE_DEPTH=4 with LEAF (shorter rollouts, more accurate leaves)
MCE_LEAF_EXPECTIMAX=1 MCE_DEPTH=4 run_bench "leaf_d4" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 750

# 5. Plain rollouts=1500 baseline (test if more rollouts alone helps)
run_bench "baseline_n1500" \
    ./target/release/cascadia-cli 50 --mce --weights $WEIGHTS --rollouts 1500

# 6. NRPA L=1 N=50 depth=6 with FAST candidates (single-level baseline)
NRPA_FAST=1 NRPA_DEPTH=6 run_bench "nrpa_l1_n50_fast" \
    ./target/release/cascadia-cli 30 --nrpa --weights $WEIGHTS --level 1 --n 50

# 7. NRPA L=2 N=8 depth=6 with FAST (smallest L=2)
NRPA_FAST=1 NRPA_DEPTH=6 run_bench "nrpa_l2_n8_fast" \
    ./target/release/cascadia-cli 30 --nrpa --weights $WEIGHTS --level 2 --n 8

echo "[$(date +%H:%M:%S)] Queue 2 complete"
