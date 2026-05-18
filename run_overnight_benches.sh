#!/bin/bash
# Sequential benchmark runner for overnight experiments
# Runs benches one at a time so they don't compete for CPU

set -e
WEIGHTS=nnue_weights_hybrid_iter4.bin
N=50

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

# 1. Combined LEAF + GUMBEL_TOPK
MCE_LEAF_EXPECTIMAX=1 MCE_GUMBEL_TOPK=1 run_bench "leaf_gumbel" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750

# 2. Combined LEAF + RANK + GUMBEL kitchen sink
MCE_LEAF_EXPECTIMAX=1 MCE_RANK_EXPECTIMAX=1 MCE_GUMBEL_TOPK=1 run_bench "leaf_rank_gumbel" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750

# 3. NRPA L=1 N=20 — fastest "real" NRPA configuration on 30 games
NRPA_DEPTH=6 run_bench "nrpa_l1_n20" \
    ./target/release/cascadia-cli 30 --nrpa --weights $WEIGHTS --level 1 --n 20

# 4. NRPA L=2 N=8 — minimal nested NRPA on 30 games
NRPA_DEPTH=6 run_bench "nrpa_l2_n8" \
    ./target/release/cascadia-cli 30 --nrpa --weights $WEIGHTS --level 2 --n 8

# 5. Gumbel-MCTS m=10 (smaller candidate pool)
run_bench "gumbel_mcts_m10" \
    ./target/release/cascadia-cli $N --gumbel-mcts --weights $WEIGHTS --rollouts 750 --m 10

# 6. MCE deeper depth (8) with leaf expectimax
MCE_DEPTH=8 MCE_LEAF_EXPECTIMAX=1 run_bench "leaf_d8" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750

# 7. MCE bigger budget (1500) with leaf expectimax
MCE_LEAF_EXPECTIMAX=1 run_bench "leaf_n1500" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 1500

# 8. Pure GUMBEL_TOPK (no leaf)
MCE_GUMBEL_TOPK=1 run_bench "gumbel_topk_only" \
    ./target/release/cascadia-cli $N --mce --weights $WEIGHTS --rollouts 750

echo "[$(date +%H:%M:%S)] All benches complete"
