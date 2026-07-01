#!/bin/bash
# Overnight bench of search improvements over the H_nnue_halving baseline.
#
# Each variant runs N games with the SAME random seed sequence so per-game
# variance is matched. Aggregate results to a summary at the end.
set -u
GAMES=${GAMES:-50}
WEIGHTS=${WEIGHTS:-nnue_weights_v9_iter14.bin}
ROLLOUTS=${ROLLOUTS:-300}
OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

run_bench() {
  local name="$1"
  shift
  local logf="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] Running $name..."
  local start=$(date +%s)
  # CLI uses deterministic 0xC0DE_C0DE seed by default — same scenarios across variants
  ./target/release/cascadia-cli "$GAMES" "$@" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

echo "═══ Search Improvements: GAMES=$GAMES, ROLLOUTS=$ROLLOUTS, WEIGHTS=$WEIGHTS ═══"
echo

# Baseline: H_nnue_halving (matches the 94.1/99.8 result)
run_bench "00_baseline_halving" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --weights "$WEIGHTS"

# Pre-filter variants
run_bench "01_prefilter_k4"  --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --prefilter-k 4 --weights "$WEIGHTS"
run_bench "02_prefilter_k6"  --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --prefilter-k 6 --weights "$WEIGHTS"
run_bench "03_prefilter_k8"  --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --prefilter-k 8 --weights "$WEIGHTS"
run_bench "04_prefilter_k12" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --prefilter-k 12 --weights "$WEIGHTS"

# Exact endgame variants (last K turns use expectimax instead of MCE)
run_bench "05_exact_endgame_1" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --exact-endgame 1 --weights "$WEIGHTS"
run_bench "06_exact_endgame_2" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --exact-endgame 2 --weights "$WEIGHTS"
run_bench "07_exact_endgame_3" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --exact-endgame 3 --weights "$WEIGHTS"

# Combined: best prefilter + exact endgame
run_bench "08_pf6_eg2" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --prefilter-k 6 --exact-endgame 2 --weights "$WEIGHTS"

echo
echo "═══ SUMMARY ═══"
printf "%-26s  %8s  %8s\n" "Variant" "Base" "Bonus"
for name in 00_baseline_halving 01_prefilter_k4 02_prefilter_k6 03_prefilter_k8 04_prefilter_k12 \
             05_exact_endgame_1 06_exact_endgame_2 07_exact_endgame_3 08_pf6_eg2; do
  f="$OUT_DIR/${name}.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-26s  %8s  %8s\n" "$name" "${base:-?}" "${bonus:-?}"
done
