#!/bin/bash
# Post-training chain for v6-peak: waits for iter 20 completion, then runs
# benchmarks comparing v6_iter20 to historical champions.
#
# Cross-binary HH (v6 vs v5sh in different binaries) is non-trivial because
# cascadia-cli runs whole games in one process with per-seat weights — and v5
# weights produce garbage when loaded into v6 binary (different feature layout).
#
# So we do TWO valid comparisons instead:
#   (A) Standard 4-player benchmark: v6_iter20 in seat 0 with mce_wide_v1
#       strategy, greedy in seats 1-3. Mean score is comparable to historical
#       CLAUDE.md numbers (champion v4opp at 95.94).
#   (B) v6 internal self-play tournament: 4× v6_iter20 self-play, mce_wide_v1
#       strategy. Tells us absolute v6 ceiling.

set -euo pipefail
LOGFILE="overnight/v6peak/post_train_chain.log"
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

log "=== v6 post-training chain started ==="

WEIGHTS="nnue_weights_v6peak_iter20.bin"
log "Step 1: waiting for $WEIGHTS"
while [ ! -f "$WEIGHTS" ]; do sleep 60; done
sleep 30  # ensure file fully written
log "  $WEIGHTS exists; v6 training complete"

# Bench A: standard 4-player vs greedy (50 games)
BENCH_A_OUT="overnight/v6peak/bench_v6_vs_greedy.log"
log "Step 2: Bench A — v6_iter20 + mce_wide_v1 vs 3× greedy (50 games)"
target-mid-v6/release/cascadia-cli 50 \
    --nnue-rollout-mce --weights "$WEIGHTS" \
    --candidates expanded --prefilter-k 8 --alloc halving --rollouts 600 \
    > "$BENCH_A_OUT" 2>&1 &
BENCH_A_PID=$!
log "  Bench A PID $BENCH_A_PID. ~50 × 145s ≈ 2 hr expected wall"

wait $BENCH_A_PID
log "  Bench A done"
log "$(grep -E 'Mean|Median|P10|P90' "$BENCH_A_OUT" | head -10)"

# Bench B: v6 internal tournament (4× v6_iter20, all in different seats with mce_wide_v1 variants)
BENCH_B_OUT="overnight/v6peak/v6_internal_tournament.jsonl"
BENCH_B_LOG="overnight/v6peak/bench_v6_internal.log"
log "Step 3: Bench B — 4-strategy tournament with v6_iter20 across all seats (50 games, parallel-1)"
python3 overnight/hh_local_v5.py \
    --strategy-a mce_wide_v1 --weights-a "$WEIGHTS" \
    --strategy-b mce_wide_v1_b --weights-b "$WEIGHTS" \
    --binary target-mid-v6/release/cascadia-cli \
    --num-games 50 --parallel 1 \
    --jsonl-out "$BENCH_B_OUT" \
    > "$BENCH_B_LOG" 2>&1
log "  Bench B done"
log "$(python3 overnight/hh_local_v5.py --summarize "$BENCH_B_OUT" 2>&1 | tail -15)"

log "=== Post-training chain complete ==="
