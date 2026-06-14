#!/bin/bash
# Follow-up bench: scale rollouts on the best variant from bench_search_improvements.
# Run this AFTER bench_search_improvements completes. Pass best variant args via $BEST_ARGS.
set -u
GAMES=${GAMES:-30}
WEIGHTS=${WEIGHTS:-nnue_weights_v9_iter14.bin}
OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

run_bench() {
  local name="$1"
  shift
  local logf="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] Running $name..."
  local start=$(date +%s)
  ./target/release/cascadia-cli "$GAMES" "$@" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

# Baseline-shape bench at different rollout counts to see if quality plateaus
echo "═══ Rollout count sweep on best variant ═══"
echo "  Variant: nnue-rollout-mce halving + prefilter k=8 + exact_endgame=2 (TBD post-bench)"

run_bench "R200_pf8_eg2" --nnue-rollout-mce --rollouts 200 --alloc halving --prefilter-k 8 --exact-endgame 2 --weights "$WEIGHTS"
run_bench "R300_pf8_eg2" --nnue-rollout-mce --rollouts 300 --alloc halving --prefilter-k 8 --exact-endgame 2 --weights "$WEIGHTS"
run_bench "R500_pf8_eg2" --nnue-rollout-mce --rollouts 500 --alloc halving --prefilter-k 8 --exact-endgame 2 --weights "$WEIGHTS"
run_bench "R750_pf8_eg2" --nnue-rollout-mce --rollouts 750 --alloc halving --prefilter-k 8 --exact-endgame 2 --weights "$WEIGHTS"

echo
echo "═══ SUMMARY ═══"
for name in R200_pf8_eg2 R300_pf8_eg2 R500_pf8_eg2 R750_pf8_eg2; do
  f="$OUT_DIR/${name}.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-26s  base=%6s  bonus=%6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
