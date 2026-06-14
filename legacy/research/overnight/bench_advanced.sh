#!/bin/bash
# Second-batch bench: more advanced variants once first batch reveals what works.
# - SeqHalvingCI (confidence-interval halving)
# - Expanded candidates + prefilter (cast wider net, focus rollouts)
# - Combinations
set -u
GAMES=${GAMES:-30}
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
  ./target/release/cascadia-cli "$GAMES" "$@" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

echo "═══ Advanced Variants: GAMES=$GAMES, ROLLOUTS=$ROLLOUTS ═══"
echo

run_bench "10_halving_ci" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving-ci --weights "$WEIGHTS"
run_bench "11_halving_ci_pf8" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving-ci --prefilter-k 8 --weights "$WEIGHTS"
run_bench "12_expanded_pf8" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --candidates expanded --prefilter-k 8 --weights "$WEIGHTS"
run_bench "13_expanded_pf12" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --candidates expanded --prefilter-k 12 --weights "$WEIGHTS"
run_bench "14_pf8_eg2_500r" --nnue-rollout-mce --rollouts 500 --alloc halving --prefilter-k 8 --exact-endgame 2 --weights "$WEIGHTS"
run_bench "15_pf8_eg2_750r" --nnue-rollout-mce --rollouts 750 --alloc halving --prefilter-k 8 --exact-endgame 2 --weights "$WEIGHTS"

echo
echo "═══ ADVANCED SUMMARY ═══"
for name in 10_halving_ci 11_halving_ci_pf8 12_expanded_pf8 13_expanded_pf12 14_pf8_eg2_500r 15_pf8_eg2_750r; do
  f="$OUT_DIR/${name}.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-26s  base=%6s  bonus=%6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
